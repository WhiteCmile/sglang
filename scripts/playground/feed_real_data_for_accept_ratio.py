#!/usr/bin/env python3
"""Feed real prompts to a running SGLang server and summarize accept ratios."""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp


def _load_local_dataset(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    prompts: List[str] = []
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = _extract_prompt(obj)
            if prompt:
                prompts.append(prompt)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for obj in data:
                prompt = _extract_prompt(obj)
                if prompt:
                    prompts.append(prompt)
        else:
            raise ValueError("JSON file must contain a top-level list.")

    if not prompts:
        raise ValueError(f"No valid prompts loaded from {path}")
    return prompts


def _extract_prompt(obj: Dict[str, Any]) -> str:
    if not isinstance(obj, dict):
        return ""

    for key in ("prompt", "input", "question", "instruction", "text"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    conversations = obj.get("conversations")
    if isinstance(conversations, list) and conversations:
        first = conversations[0]
        if isinstance(first, dict):
            val = first.get("value")
            if isinstance(val, str) and val.strip():
                return val.strip()

    messages = obj.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()

    return ""


def _load_hf_ultrachat(max_samples: int) -> List[str]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets is required for --dataset-source ultrachat. "
            "Install with: pip install datasets"
        ) from e

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    prompts: List[str] = []
    for row in ds:
        messages = row.get("messages", [])
        if not isinstance(messages, list):
            continue
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    prompts.append(content.strip())
                    break
        if len(prompts) >= max_samples:
            break
    if not prompts:
        raise ValueError("No prompts loaded from ultrachat.")
    return prompts


def _extract_meta_info(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, dict):
        meta_info = payload.get("meta_info")
        if isinstance(meta_info, dict):
            return meta_info
        for value in payload.values():
            found = _extract_meta_info(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _extract_meta_info(item)
            if found is not None:
                return found
    return None


async def _send_one(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.perf_counter()
    async with session.post(url, json=payload, timeout=timeout_s) as resp:
        text = await resp.text()
        latency = time.perf_counter() - t0
        if resp.status != 200:
            return {"ok": False, "status": resp.status, "latency": latency, "body": text}
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            return {
                "ok": False,
                "status": 200,
                "latency": latency,
                "body": text,
                "error": "invalid_json",
            }
        meta_info = _extract_meta_info(body)
        return {
            "ok": True,
            "status": 200,
            "latency": latency,
            "body": body,
            "meta_info": meta_info,
        }


def _format_ratio(values: List[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"count={len(values)} "
        f"mean={statistics.fmean(values):.4f} "
        f"min={min(values):.4f} "
        f"p50={statistics.median(values):.4f} "
        f"max={max(values):.4f}"
    )


async def run(args: argparse.Namespace) -> None:
    if args.dataset_source == "local":
        prompts = _load_local_dataset(Path(args.dataset_path))
    else:
        prompts = _load_hf_ultrachat(args.num_batches * args.batch_size * 2)

    random.seed(args.seed)
    random.shuffle(prompts)
    need = args.num_batches * args.batch_size
    if len(prompts) < need:
        prompts = (prompts * ((need + len(prompts) - 1) // len(prompts)))[:need]
    else:
        prompts = prompts[:need]

    url = args.base_url.rstrip("/") + "/chat/completions"
    print(f"Sending to {url}")
    print(
        f"dataset_source={args.dataset_source} total_prompts={len(prompts)} "
        f"num_batches={args.num_batches} batch_size={args.batch_size}"
    )

    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    connector = aiohttp.TCPConnector(limit=max(args.max_concurrency, args.batch_size))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        ptr = 0
        total_ok = 0
        total_fail = 0
        missing_meta = 0
        latencies: List[float] = []
        spec_accept_rate_values: List[float] = []
        spec_accept_rate_per_step_values: List[float] = []
        spec_accept_rate_per_draft_token_values: List[float] = []

        for batch_idx in range(args.num_batches):
            batch_prompts = prompts[ptr : ptr + args.batch_size]
            ptr += args.batch_size
            tasks = [
                _send_one(
                    session=session,
                    url=url,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout_s=args.timeout_s,
                )
                for prompt in batch_prompts
            ]

            sem = asyncio.Semaphore(args.max_concurrency)

            async def _guard(coro):
                async with sem:
                    return await coro

            results = await asyncio.gather(*[_guard(task) for task in tasks])
            ok = sum(1 for result in results if result["ok"])
            fail = len(results) - ok
            total_ok += ok
            total_fail += fail
            latencies.extend([result["latency"] for result in results if result["ok"]])

            batch_meta = 0
            for result in results:
                if not result["ok"]:
                    continue
                meta_info = result.get("meta_info")
                if not isinstance(meta_info, dict):
                    missing_meta += 1
                    continue
                batch_meta += 1

                value = meta_info.get("spec_accept_rate")
                if isinstance(value, (int, float)):
                    spec_accept_rate_values.append(float(value))

                value = meta_info.get("spec_accept_rate_per_step")
                if isinstance(value, (int, float)):
                    spec_accept_rate_per_step_values.append(float(value))

                value = meta_info.get("spec_accept_rate_per_draft_token")
                if isinstance(value, (int, float)):
                    spec_accept_rate_per_draft_token_values.append(float(value))

            avg_latency = sum(result["latency"] for result in results) / len(results)
            print(
                f"batch={batch_idx:04d} ok={ok}/{len(results)} fail={fail} "
                f"meta_info={batch_meta}/{ok} avg_latency={avg_latency:.3f}s"
            )

        if latencies:
            print(
                f"done total_ok={total_ok} total_fail={total_fail} "
                f"missing_meta={missing_meta} avg_ok_latency={sum(latencies)/len(latencies):.3f}s"
            )
        else:
            print(
                f"done total_ok={total_ok} total_fail={total_fail} missing_meta={missing_meta}"
            )

        print("spec_accept_rate:", _format_ratio(spec_accept_rate_values))
        print(
            "spec_accept_rate_per_step:",
            _format_ratio(spec_accept_rate_per_step_values),
        )
        print(
            "spec_accept_rate_per_draft_token:",
            _format_ratio(spec_accept_rate_per_draft_token_values),
        )

        if args.output_json:
            summary = {
                "total_ok": total_ok,
                "total_fail": total_fail,
                "missing_meta": missing_meta,
                "spec_accept_rate": spec_accept_rate_values,
                "spec_accept_rate_per_step": spec_accept_rate_per_step_values,
                "spec_accept_rate_per_draft_token": (
                    spec_accept_rate_per_draft_token_values
                ),
            }
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"wrote summary json to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:30001/v1",
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument(
        "--dataset-source",
        type=str,
        choices=["local", "ultrachat"],
        default="local",
        help="Use local json/jsonl dataset, or load ultrachat from HuggingFace.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="experiments/data/sharegpt.json",
        help="Path to local json/jsonl dataset when dataset-source=local.",
    )
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to write raw accept-ratio samples as JSON.",
    )
    args = parser.parse_args()

    if args.dataset_source == "local" and not args.dataset_path:
        raise ValueError("--dataset-path is required when --dataset-source=local")
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be > 0")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
