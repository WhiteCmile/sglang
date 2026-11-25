"""bf16 vs FP8 Target quality: GSM8K (5-shot CoT, all 1319) and MMLU (5-shot, all 14042) on a plain
SGLang server, greedy. Usage:
    eval_quality.py --url http://127.0.0.1:30000 --out <dir> [--tasks gsm8k,mmlu] [--parallel 128]
Writes <dir>/<task>.jsonl (per item: id, pred, gold, correct, text) and <dir>/summary.json.
Speculative decoding verifies with the Target's own greedy argmax, so the Target model alone decides output
quality; a plain server isolates the weight format (bf16 vs online FP8).
"""
import argparse
import concurrent.futures as cf
import csv
import glob
import json
import os
import re
import time
import urllib.request

import pandas as pd

GSM8K_DIR = "/data/data1/datasets/openai/gsm8k/main"
MMLU_DIR = "/data/data1/opencompass/data/mmlu"
REQUEST_TIMEOUT_S = 600
MAX_ATTEMPTS = 3


def generate(url, text, max_new_tokens, stop=None):
    body = {"text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens}}
    if stop:
        body["sampling_params"]["stop"] = stop
    data = json.dumps(body).encode()
    for attempt in range(MAX_ATTEMPTS):
        try:
            req = urllib.request.Request(url + "/generate", data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
                return json.loads(r.read())["text"]
        except Exception:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)


def gsm8k_items():
    train = pd.read_parquet(f"{GSM8K_DIR}/train-00000-of-00001.parquet")
    test = pd.read_parquet(f"{GSM8K_DIR}/test-00000-of-00001.parquet")
    shots = "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a in zip(train.question[:5], train.answer[:5]))
    for i, (q, a) in enumerate(zip(test.question, test.answer)):
        gold = a.split("####")[-1].strip().replace(",", "")
        yield {"id": i, "prompt": shots + f"Question: {q}\nAnswer:", "gold": gold}


def gsm8k_pred(text):
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    nums = [m.group(1)] if m else re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not nums:
        return None
    v = nums[-1].replace(",", "").rstrip(".")
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return v


def mmlu_items():
    def fmt(row):
        return f"{row[0]}\nA. {row[1]}\nB. {row[2]}\nC. {row[3]}\nD. {row[4]}\nAnswer:"
    for path in sorted(glob.glob(f"{MMLU_DIR}/test/*_test.csv")):
        subject = os.path.basename(path)[: -len("_test.csv")]
        dev = list(csv.reader(open(f"{MMLU_DIR}/dev/{subject}_dev.csv")))[:5]
        head = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
        shots = "".join(fmt(r) + f" {r[5]}\n\n" for r in dev)
        for j, row in enumerate(csv.reader(open(path))):
            yield {"id": f"{subject}/{j}", "subject": subject, "prompt": head + shots + fmt(row), "gold": row[5].strip()}


TASKS = {
    "gsm8k": (gsm8k_items, lambda u, p: generate(u, p, 512, ["Question:", "\n\n\n"]), gsm8k_pred),
    "mmlu": (mmlu_items, lambda u, p: generate(u, p, 1), lambda t: t.strip()[:1].upper() or None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="gsm8k,mmlu")
    ap.add_argument("--parallel", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summary = {}
    for task in a.tasks.split(","):
        items_fn, gen_fn, pred_fn = TASKS[task]
        items = list(items_fn())
        t0 = time.time()
        with cf.ThreadPoolExecutor(a.parallel) as ex:
            texts = list(ex.map(lambda it: gen_fn(a.url, it["prompt"]), items))
        correct = 0
        with open(f"{a.out}/{task}.jsonl", "w") as f:
            for it, text in zip(items, texts):
                pred = pred_fn(text)
                ok = pred is not None and pred == it["gold"]
                correct += ok
                f.write(json.dumps({"id": it["id"], "pred": pred, "gold": it["gold"], "correct": ok, "text": text}) + "\n")
        summary[task] = {"n": len(items), "correct": correct, "accuracy": correct / len(items), "seconds": round(time.time() - t0, 1)}
        print(task, summary[task], flush=True)
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
