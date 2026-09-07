#!/usr/bin/env python3
"""Run cold-started H11 baseline/DT algorithm cells with strict graph checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import shutil
import socket
import stat
import subprocess
import time
import urllib.request
from pathlib import Path


WORKSPACE = Path("/share/dai-sys/zhaotianlang/draft-target-disaggregation")
RUNTIME = WORKSPACE / "dt-runtime-algo"
SGLANG = WORKSPACE / "sglang-dt-algo"
PYTHON = WORKSPACE / "sglang-v1-clean/.venv/bin/python"
PROFILE = RUNTIME / "configs/qwen3-14b-flashmtp-v2.toml"
PROMPTS = WORKSPACE / "benchmark-data/flashmtp-prompts-20260731/specbench_reasoning.json"
PROMPTS_SHA256 = "bc3baa1ac11f9e832fff50bb5348d57e3effaf70e060c6457dbac6b95cac2b1f"
GRAPH_RE = re.compile(
    r'^sglang:cuda_graph_passes_total\{([^}]*)\}\s+([0-9.eE+-]+)$'
)
FATAL_RE = re.compile(
    r"Traceback \(most recent call last\)|RuntimeError:|CUDA error:|"
    r"NCCL[^\n]*(?:error|fail)|fatal[^\n]*error|replay[^\n]*(?:error|fail)",
    re.I,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def command_output(argv: list[str]) -> str:
    return subprocess.run(
        argv, check=True, capture_output=True, text=True
    ).stdout.strip()


def git_state(repo: Path, artifact_dir: Path, name: str) -> dict[str, object]:
    """Fingerprint tracked edits and exact untracked source bytes."""
    head = command_output(
        [
            "git",
            "-c",
            f"safe.directory={repo}",
            "-C",
            str(repo),
            "rev-parse",
            "HEAD",
        ]
    )
    diff = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo}",
            "-C",
            str(repo),
            "diff",
            "--binary",
            "HEAD",
        ],
        check=True,
        capture_output=True,
    ).stdout
    (artifact_dir / f"{name}-tracked.patch").write_bytes(diff)
    untracked_raw = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo}",
            "-C",
            str(repo),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    ).stdout
    untracked_paths = [
        Path(value.decode("utf-8")) for value in untracked_raw.split(b"\0") if value
    ]
    untracked = []
    state_hash = hashlib.sha256(diff)
    for relative_path in sorted(untracked_paths):
        payload = (repo / relative_path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        untracked.append(
            {
                "path": relative_path.as_posix(),
                "sha256": digest,
                "size": len(payload),
            }
        )
        state_hash.update(relative_path.as_posix().encode("utf-8") + b"\0")
        state_hash.update(bytes.fromhex(digest))
    atomic_json(artifact_dir / f"{name}-untracked.json", untracked)
    return {
        "head": head,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked": untracked,
        "state_sha256": state_hash.hexdigest(),
    }


def gpu_apps() -> list[dict[str, object]]:
    raw = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, object]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        uuid, pid, name, memory = [part.strip() for part in line.split(",", 3)]
        rows.append(
            {"gpu_uuid": uuid, "pid": int(pid), "name": name, "memory_mib": memory}
        )
    return rows


def listening_ports() -> set[int]:
    ports: set[int] = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 4 and fields[3] == "0A":
                ports.add(int(fields[1].rsplit(":", 1)[1], 16))
    return ports


def fixed_ports() -> set[int]:
    return set(range(30000, 30003)) | set(range(31000, 31004)) | {
        19001,
        19002,
    }


def proc_identity(pid: int) -> dict[str, object]:
    stat_path = Path(f"/proc/{pid}/stat")
    raw = stat_path.read_text()
    tail = raw[raw.rfind(")") + 2 :].split()
    return {
        "pid": pid,
        "pgid": int(tail[2]),
        "starttime": int(tail[19]),
        "uid": os.stat(f"/proc/{pid}").st_uid,
    }


def same_process(identity: dict[str, object]) -> bool:
    try:
        return proc_identity(int(identity["pid"])) == identity
    except (FileNotFoundError, ProcessLookupError):
        return False


def endpoints(topology: str) -> list[tuple[int, str]]:
    ports = range(31000, 31004) if topology == "baseline" else range(30000, 30003)
    return [(port, f"http://127.0.0.1:{port}") for port in ports]


def healthy(endpoint: str) -> bool:
    for suffix in ("/health_generate", "/health"):
        try:
            with urllib.request.urlopen(endpoint + suffix, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
    return False


def wait_healthy(
    launcher: subprocess.Popen[str], topology: str, timeout_s: float
) -> None:
    pending = {endpoint for _, endpoint in endpoints(topology)}
    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
        require(launcher.poll() is None, f"{topology} launcher exited during startup")
        pending = {endpoint for endpoint in pending if not healthy(endpoint)}
        if pending:
            time.sleep(3)
    require(not pending, f"unhealthy endpoints: {sorted(pending)}")


def metric_text(endpoint: str) -> str:
    with urllib.request.urlopen(endpoint + "/metrics", timeout=10) as response:
        require(response.status == 200, f"metrics HTTP {response.status}")
        return response.read().decode("utf-8")


def graph_modes(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = GRAPH_RE.match(line)
        if match is None:
            continue
        labels, raw = match.groups()
        mode_match = re.search(r'(?:^|,)mode="([^"]+)"', labels)
        require(mode_match is not None, "CUDA Graph metric lacks mode")
        mode = mode_match.group(1)
        value = float(raw)
        require(math.isfinite(value) and value >= 0, "invalid CUDA Graph counter")
        values[mode] = values.get(mode, 0.0) + value
    return values


def graph_delta(before: str, after: str) -> dict[str, float]:
    left, right = graph_modes(before), graph_modes(after)
    modes = set(left) | set(right)
    require(modes, "missing CUDA Graph metrics")
    delta = {mode: right.get(mode, 0.0) - left.get(mode, 0.0) for mode in modes}
    require(all(value >= 0 for value in delta.values()), "graph counter regression")
    require(delta.get("decode_cuda_graph", 0.0) > 0, "no decode graph executions")
    require(delta.get("prefill_cuda_graph", 0.0) > 0, "no prefill graph executions")
    require(delta.get("decode_none", 0.0) == 0, "decode graph fallback")
    require(delta.get("prefill_none", 0.0) == 0, "prefill graph fallback")
    return dict(sorted(delta.items()))


def capture_audit(leg: Path, topology: str) -> dict[str, object]:
    names = (
        [f"baseline-{index}" for index in range(4)]
        if topology == "baseline"
        else [f"target-{index}" for index in range(3)]
    )
    report: dict[str, object] = {}
    for name in names:
        text = (leg / "servers" / f"{name}.log").read_text(errors="replace")
        normalized = text.replace("\r", "\n")
        require(normalized.count("Capture target prefill CUDA graph begin.") == 2, f"{name}: prefill capture")
        require(normalized.count("Capture target verify CUDA graph begin.") == 2, f"{name}: verify capture")
        require("61/61" in normalized and "59/59" in normalized, f"{name}: incomplete target capture")
        require("disable_cuda_graph=False" in normalized, f"{name}: graph disabled")
        require("disable_cuda_graph_padding=False" in normalized, f"{name}: graph padding disabled")
        if topology == "baseline":
            require(normalized.count("Capture draft verify CUDA graph begin.") == 2, f"{name}: colocated draft capture")
        report[name] = {"target_prefill_buckets": 61, "target_decode_buckets": 59}
    if topology == "dt":
        for name in ("draft-0", "draft-1"):
            text = (leg / "servers" / f"{name}.log").read_text(errors="replace").replace("\r", "\n")
            require("Captured standalone Draft CUDA graphs:" in text, f"{name}: missing draft capture")
            require("batch_sizes=(1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128)" in text, f"{name}: wrong draft capture grid")
            require("cuda_graph=True" in text, f"{name}: draft graph disabled")
            report[name] = {"draft_decode_buckets": 36}
    return report


def fatal_audit(leg: Path) -> None:
    failures = []
    for path in sorted((leg / "servers").glob("*.log")):
        if FATAL_RE.search(path.read_text(errors="replace")):
            failures.append(path.name)
    require(not failures, f"fatal server logs: {failures}")


def stop_owned_launcher(
    launcher: subprocess.Popen[str], identity: dict[str, object]
) -> None:
    if launcher.poll() is None and same_process(identity):
        os.killpg(launcher.pid, signal.SIGTERM)
    try:
        launcher.wait(timeout=120)
    except subprocess.TimeoutExpired:
        if same_process(identity):
            os.killpg(launcher.pid, signal.SIGKILL)
        launcher.wait(timeout=20)


def wait_cold(timeout_s: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        busy_ports = fixed_ports() & listening_ports()
        apps = gpu_apps()
        if not busy_ports and not apps:
            return
        time.sleep(2)
    raise RuntimeError(
        f"resources did not become cold: ports={sorted(busy_ports)}, apps={apps}"
    )


def remove_owned_dt_data_dir(path: Path) -> None:
    require(path.parent == Path("/dev/shm"), f"unsafe DT data parent: {path}")
    require(path.name.startswith("dt-algo-dt-algo-"), f"unsafe DT data name: {path}")
    if not path.exists():
        return
    uid = os.getuid()
    root_stat = os.lstat(path)
    require(stat.S_ISDIR(root_stat.st_mode), f"DT data root is not a directory: {path}")
    require(root_stat.st_uid == uid, f"foreign DT data root: {path}")
    for current, dirs, files in os.walk(path, topdown=True, followlinks=False):
        for name in dirs + files:
            child = Path(current) / name
            child_stat = os.lstat(child)
            require(child_stat.st_uid == uid, f"foreign DT data entry: {child}")
            require(not stat.S_ISLNK(child_stat.st_mode), f"symlink in DT data: {child}")
    shutil.rmtree(path)


def validate_result(path: Path, args: argparse.Namespace, topology: str) -> dict:
    result = json.loads(path.read_text())
    expected_tokens = args.measurement_requests * args.max_new_tokens
    expected = {
        "topology": topology,
        "global_concurrency": args.concurrency,
        "routing_policy": "least-inflight",
        "warmup_requests": 32,
        "measurement_requests": args.measurement_requests,
        "max_new_tokens": args.max_new_tokens,
        "seed": 0,
        "ignore_eos": True,
        "prompts_sha256": args.prompts_sha256,
    }
    for key, value in expected.items():
        require(result.get(key) == value, f"result {key}: {result.get(key)!r} != {value!r}")
    require(result.get("fixed_work_valid") is True, "fixed work invalid")
    require(result.get("successful_requests") == args.measurement_requests, "request failures")
    require(result.get("failed_requests") == 0, "request failures")
    require(result.get("committed_output_tokens") == expected_tokens, "token total")
    records = result.get("records")
    require(isinstance(records, list) and len(records) == args.measurement_requests, "record count")
    output_digest = hashlib.sha256(
        json.dumps(
            [record["output_token_ids_sha256"] for record in records],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "throughput": result["committed_output_tokens_per_s"],
        "wall_time_s": result["wall_time_s"],
        "spec_accept_length": result["spec_accept_length"],
        "spec_verify_ct": result["spec_verify_ct"],
        "output_digest": output_digest,
        "result_sha256": sha256(path),
    }


def run_leg(args: argparse.Namespace, label: str, ordinal: int) -> dict:
    topology = "baseline" if label.startswith("b") else "dt"
    algo = label.endswith("1")
    if args.factor == "bundle":
        budgeted_verify = algo
        ngram_correction = algo
    elif args.factor == "ngram":
        budgeted_verify = True
        ngram_correction = algo
    else:
        budgeted_verify = algo
        ngram_correction = False
    leg = args.output_dir / f"leg-{ordinal}-{label}"
    leg.mkdir()
    server_dir = leg / "servers"
    dt_data_dir = Path(f"/dev/shm/dt-algo-{args.output_dir.name}-leg{ordinal}-{label}")
    require(not dt_data_dir.exists(), f"stale DT data dir: {dt_data_dir}")
    wait_cold()
    env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "LANG": "C.UTF-8",
        "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
        "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
        "PYTHONPATH": str(RUNTIME / "src"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "SPT_NOENV": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "SGLANG_FLASHMTP_SHARED_OPTIMIZATIONS": "0",
        "SGLANG_FLASHMTP_RAGGED_VERIFY": "0",
        "SGLANG_FLASHMTP_ADAPTIVE_VERIFY": "0",
        "SGLANG_FLASHMTP_PINNED_VERIFY_METADATA": "0",
        "SGLANG_FLASHMTP_VOCAB_PARALLEL_GREEDY": "0",
        "SGLANG_FLASHMTP_GRAPH_TIER_PADDING": "0",
        "SGLANG_FLASHMTP_BUDGETED_VERIFY": "1" if budgeted_verify else "0",
        "SGLANG_FLASHMTP_NGRAM_CORRECTION": (
            "1" if ngram_correction else "0"
        ),
        "SGLANG_FLASHMTP_NGRAM_PROMPT_AFFINITY": (
            "1" if args.ngram_prompt_affinity else "0"
        ),
    }
    if ngram_correction:
        env["SGLANG_FLASHMTP_NGRAM_HORIZON"] = str(args.ngram_horizon)
        env["SGLANG_FLASHMTP_NGRAM_MAX_ENTRIES"] = str(args.ngram_max_entries)
        env["SGLANG_FLASHMTP_NGRAM_MIN_COUNT"] = str(args.ngram_min_count)
        env["SGLANG_FLASHMTP_NGRAM_MIN_SOURCES"] = str(args.ngram_min_sources)
        env["SGLANG_FLASHMTP_NGRAM_ORDER"] = str(args.ngram_order)
        env["SGLANG_FLASHMTP_NGRAM_SOURCE_GATED_FROM"] = str(
            args.ngram_source_gated_from
        )
        env["SGLANG_FLASHMTP_NGRAM_SHARED_PROBE_INTERVAL"] = str(
            args.ngram_shared_probe_interval
        )
        if args.vocab_parallel_greedy:
            env["SGLANG_FLASHMTP_VOCAB_PARALLEL_GREEDY"] = "1"
    if args.thresholds:
        env["SGLANG_FLASHMTP_BUDGETED_VERIFY_THRESHOLDS"] = args.thresholds
    if args.enable_profile:
        env["DT_PROFILE_DIR"] = str(leg / "profile")
    launcher_argv = [
        str(PYTHON), "-m", "dt_launcher", topology,
        "--preset", "h11-production-v1",
        "--workspace", str(WORKSPACE),
        "--python-executable", str(PYTHON),
        "--profile", str(PROFILE),
        "--dt-stage", "3t2d",
        "--dt-data-dir", str(dt_data_dir),
        "--run-dir", str(server_dir),
        "--dt-sglang-dir", str(SGLANG),
        "--baseline-sglang-dir", str(SGLANG),
    ]
    result_path = leg / "result.json"
    benchmark_argv = [
        str(PYTHON), "-m", "dt_benchmark",
        "--topology", topology,
        "--prompts-file", str(args.prompts_file),
        "--output", str(result_path),
        "--global-concurrency", str(args.concurrency),
        "--routing-policy", "least-inflight",
        "--warmup-requests", "32",
        "--measurement-requests", str(args.measurement_requests),
        "--max-new-tokens", str(args.max_new_tokens),
        "--seed", "0", "--ignore-eos",
        "--request-timeout-s", "900", "--health-timeout-s", "1800",
    ]
    atomic_json(
        leg / "invocation.json",
        {
            "label": label,
            "topology": topology,
            "algorithm": algo,
            "factor": args.factor,
            "budgeted_verify": budgeted_verify,
            "ngram_correction": ngram_correction,
            "launcher_argv": launcher_argv,
            "benchmark_argv": benchmark_argv,
            "environment": env,
        },
    )
    launcher: subprocess.Popen[str] | None = None
    identity: dict[str, object] | None = None
    before: dict[int, str] = {}
    after: dict[int, str] = {}
    try:
        with (leg / "launcher.log").open("x", encoding="utf-8") as launcher_log:
            launcher = subprocess.Popen(
                launcher_argv,
                cwd=RUNTIME,
                env=env,
                stdout=launcher_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            identity = proc_identity(launcher.pid)
            require(identity["pgid"] == launcher.pid, "launcher is not group leader")
            wait_healthy(launcher, topology, args.startup_timeout_s)
            capture = capture_audit(leg, topology)
            fatal_audit(leg)
            require(len(gpu_apps()) == 8, "expected exactly eight GPU processes")
            for port, endpoint in endpoints(topology):
                before[port] = metric_text(endpoint)
                (leg / f"metrics-before-{port}.prom").write_text(before[port])
            with (leg / "benchmark.log").open("x", encoding="utf-8") as benchmark_log:
                subprocess.run(
                    benchmark_argv,
                    cwd=RUNTIME,
                    env=env,
                    stdout=benchmark_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True,
                    timeout=args.run_timeout_s,
                )
            for port, endpoint in endpoints(topology):
                after[port] = metric_text(endpoint)
                (leg / f"metrics-after-{port}.prom").write_text(after[port])
            metrics = {
                str(port): graph_delta(before[port], after[port]) for port in before
            }
            result = validate_result(result_path, args, topology)
            fatal_audit(leg)
            audit = {"capture": capture, "metrics": metrics, "result": result}
            atomic_json(leg / "audit.json", audit)
            return {
                "label": label,
                "topology": topology,
                "algorithm": algo,
                "factor": args.factor,
                "budgeted_verify": budgeted_verify,
                "ngram_correction": ngram_correction,
                **result,
            }
    finally:
        if launcher is not None and identity is not None:
            stop_owned_launcher(launcher, identity)
        wait_cold()
        remove_owned_dt_data_dir(dt_data_dir)
        require(not dt_data_dir.exists(), f"launcher left DT data dir: {dt_data_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, default=PROMPTS)
    parser.add_argument("--require-unique-measurement-prompts", action="store_true")
    parser.add_argument("--schedule", default="b1,d1,b0,d0")
    parser.add_argument(
        "--factor",
        choices=("bundle", "ngram", "budgeted_verify"),
        default="bundle",
    )
    parser.add_argument("--concurrency", type=int, default=768)
    parser.add_argument("--measurement-requests", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--ngram-min-count", type=int, default=1)
    parser.add_argument("--ngram-min-sources", type=int, default=2)
    parser.add_argument("--ngram-order", type=int, default=12)
    parser.add_argument("--ngram-horizon", type=int, default=12)
    parser.add_argument("--ngram-max-entries", type=int, default=262144)
    parser.add_argument("--ngram-shared-probe-interval", type=int, default=1)
    parser.add_argument("--ngram-source-gated-from", type=int, default=0)
    parser.add_argument("--ngram-prompt-affinity", action="store_true")
    parser.add_argument("--vocab-parallel-greedy", action="store_true")
    parser.add_argument("--enable-profile", action="store_true")
    parser.add_argument("--startup-timeout-s", type=float, default=1800)
    parser.add_argument("--run-timeout-s", type=float, default=7200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(socket.gethostname() == "is-dcszf4wke5colmrw-devmachine-0", "wrong host")
    require(args.concurrency == 768, "this runner is fixed to B768")
    require(args.max_new_tokens == 320, "this runner is fixed to max_new_tokens=320")
    require(args.ngram_min_count > 0, "ngram min count must be positive")
    require(args.ngram_min_sources > 0, "ngram min sources must be positive")
    require(args.ngram_order > 0, "ngram order must be positive")
    require(args.ngram_horizon >= 8, "ngram horizon must be at least 8")
    require(args.ngram_max_entries > 0, "ngram max entries must be positive")
    require(
        0 <= args.ngram_source_gated_from < args.ngram_horizon,
        "ngram source gate must be within proposal width",
    )
    require(
        args.ngram_shared_probe_interval > 0,
        "ngram shared probe interval must be positive",
    )
    labels = [value.strip() for value in args.schedule.split(",") if value.strip()]
    require(labels and set(labels) <= {"b0", "b1", "d0", "d1"}, "bad schedule")
    require(not args.output_dir.exists(), "output directory already exists")
    args.prompts_file = args.prompts_file.resolve()
    require(args.prompts_file.is_file(), "prompts file does not exist")
    args.prompts_sha256 = sha256(args.prompts_file)
    if args.prompts_file == PROMPTS:
        require(args.prompts_sha256 == PROMPTS_SHA256, "prompt hash mismatch")
    prompts = json.loads(args.prompts_file.read_text(encoding="utf-8"))
    require(
        isinstance(prompts, list)
        and prompts
        and all(isinstance(prompt, str) and prompt for prompt in prompts),
        "prompts file must be a non-empty JSON list of strings",
    )
    if args.require_unique_measurement_prompts:
        warmup_requests = 32
        measured = [
            prompts[(warmup_requests + index) % len(prompts)]
            for index in range(args.measurement_requests)
        ]
        warmup = [prompts[index % len(prompts)] for index in range(warmup_requests)]
        require(
            len(set(measured)) == args.measurement_requests,
            "measurement prompts are not unique",
        )
        require(
            set(warmup).isdisjoint(measured),
            "warmup and measurement prompts overlap",
        )
    wait_cold(timeout_s=1)
    args.output_dir.mkdir(parents=True)
    sglang_state = git_state(SGLANG, args.output_dir, "sglang")
    runtime_state = git_state(RUNTIME, args.output_dir, "runtime")
    provenance = {
        "host": socket.gethostname(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sglang": sglang_state,
        "runtime": runtime_state,
        "prompts_file": str(args.prompts_file),
        "prompts_sha256": args.prompts_sha256,
        "profile_sha256": sha256(PROFILE),
        "schedule": labels,
        "concurrency": args.concurrency,
        "measurement_requests": args.measurement_requests,
        "max_new_tokens": args.max_new_tokens,
        "thresholds": args.thresholds or None,
        "factor": args.factor,
        "ngram_min_count": args.ngram_min_count,
        "ngram_min_sources": args.ngram_min_sources,
        "ngram_order": args.ngram_order,
        "ngram_horizon": args.ngram_horizon,
        "ngram_max_entries": args.ngram_max_entries,
        "ngram_shared_probe_interval": args.ngram_shared_probe_interval,
        "ngram_source_gated_from": args.ngram_source_gated_from,
        "ngram_prompt_affinity": args.ngram_prompt_affinity,
        "vocab_parallel_greedy": args.vocab_parallel_greedy,
    }
    atomic_json(args.output_dir / "provenance.json", provenance)
    results = []
    try:
        for ordinal, label in enumerate(labels, 1):
            print(f"START leg={ordinal} label={label}", flush=True)
            result = run_leg(args, label, ordinal)
            results.append(result)
            atomic_json(args.output_dir / "partial-summary.json", {"results": results})
            print(json.dumps(result, sort_keys=True), flush=True)
    except BaseException as exc:
        atomic_json(
            args.output_dir / "failure.json",
            {"type": type(exc).__name__, "message": str(exc), "results": results},
        )
        raise
    by_label = {result["label"]: result for result in results}
    comparisons: dict[str, object] = {}
    for algo in (0, 1):
        baseline, dt = by_label.get(f"b{algo}"), by_label.get(f"d{algo}")
        if baseline and dt:
            comparisons[f"dt_over_baseline_algo_{algo}"] = (
                dt["throughput"] / baseline["throughput"]
            )
    if "b0" in by_label and "b1" in by_label:
        comparisons["baseline_algo_lift"] = by_label["b1"]["throughput"] / by_label["b0"]["throughput"]
    if "d0" in by_label and "d1" in by_label:
        comparisons["dt_algo_lift"] = by_label["d1"]["throughput"] / by_label["d0"]["throughput"]
    comparisons["all_output_digests_equal"] = len({result["output_digest"] for result in results}) == 1
    summary = {"provenance": provenance, "results": results, "comparisons": comparisons}
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
