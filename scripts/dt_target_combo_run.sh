#!/bin/bash
set -euo pipefail

W=${DT_EXPERIMENT_WORKSPACE:-/data/lishuhuan/dt-v3-pilot}
SOURCE=${DT_EXPERIMENT_SOURCE:-/data/lishuhuan/dt-v3-page}
stamp=${1:-$(date +%Y%m%d-%H%M%S)-target-combo}
root="$W/target-combo-results/$stamp"
arms=(control early_slots8 early_page64 early_page64_slots8)
topology="$root/topology.toml"
local_ports=()
remote_ports=()

mkdir -p "$root"
printf '%s\n' "$stamp" >"$root/stamp.txt"
git -C "$SOURCE" rev-parse HEAD >"$root/source-commit.txt"
printf '%s\n' "${arms[@]}" >"$root/arms.txt"

service_pid=
benchmark_pid=
cleanup_service() {
  if [ -n "${service_pid:-}" ] && kill -0 "$service_pid" 2>/dev/null; then
    kill -INT -- "-$service_pid" 2>/dev/null || true
    for _ in $(seq 1 90); do
      kill -0 "$service_pid" 2>/dev/null || break
      sleep 2
    done
    if kill -0 "$service_pid" 2>/dev/null; then
      kill -TERM -- "-$service_pid" 2>/dev/null || true
    fi
  fi
  if [ -n "${service_pid:-}" ]; then
    wait "$service_pid" 2>/dev/null || true
  fi
  service_pid=
}

cleanup_benchmark() {
  if [ -n "${benchmark_pid:-}" ] && kill -0 "$benchmark_pid" 2>/dev/null; then
    kill -TERM "$benchmark_pid" 2>/dev/null || true
  fi
  if [ -n "${benchmark_pid:-}" ]; then
    wait "$benchmark_pid" 2>/dev/null || true
  fi
  benchmark_pid=
}

load_topology_ports() {
  mapfile -t local_ports < <("$SOURCE/python-local" - "$topology" sys-3 <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as f:
    topology = tomllib.load(f)
host = sys.argv[2]
coordinator = topology["coordinator"]
if coordinator["host"] == host:
    print(coordinator["target_port"])
    print(coordinator["draft_port"])
for instance in topology["instances"]:
    if instance["host"] != host:
        continue
    for key in ("http_port", "nccl_port", "tcp_port"):
        if key in instance:
            print(instance[key])
PY
)
  mapfile -t remote_ports < <("$SOURCE/python-local" - "$topology" sys-4 <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as f:
    topology = tomllib.load(f)
host = sys.argv[2]
for instance in topology["instances"]:
    if instance["host"] != host:
        continue
    for key in ("http_port", "nccl_port", "tcp_port"):
        if key in instance:
            print(instance[key])
PY
)
}

probe_remote_ports() {
  ssh -o BatchMode=yes -o ConnectTimeout=8 \
      "${DT_SYS4_SSH_HOST:-h8004-lsh-target-ab}" \
      python3 - "${remote_ports[@]}" <<'PY'
import socket
import sys

busy = []
for raw_port in sys.argv[1:]:
    port = int(raw_port)
    sock = socket.socket()
    try:
        sock.bind(("0.0.0.0", port))
    except OSError as exc:
        busy.append((port, str(exc)))
    finally:
        sock.close()
if busy:
    raise SystemExit(f"remote ports unavailable: {busy}")
PY
}

local_gpus_idle() {
  ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'
}

remote_gpus_idle() {
  ssh -o BatchMode=yes -o ConnectTimeout=8 \
      "${DT_SYS4_SSH_HOST:-h8004-lsh-target-ab}" \
      "! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'"
}

wait_for_dual_host_gpu_idle() {
  local deadline=$((SECONDS + ${DT_GPU_IDLE_WAIT_S:-21600}))
  local idle_polls=0
  while [ "$SECONDS" -lt "$deadline" ]; do
    if local_gpus_idle && remote_gpus_idle; then
      idle_polls=$((idle_polls + 1))
      if [ "$idle_polls" -ge 3 ]; then
        return 0
      fi
    else
      idle_polls=0
    fi
    sleep 5
  done
  echo "both DT hosts did not become GPU-idle before the wait deadline" >&2
  return 1
}

allocate_topology() {
  local attempt
  local base
  for attempt in $(seq 1 64); do
    base=$((24000 + ((RANDOM * 32768 + RANDOM) % 30000)))
    "$SOURCE/python-local" "$SOURCE/scripts/prepare_dt_combo_topology.py" \
      --template "$W/topology-7t2d-page-ab.toml" \
      --output "$topology" --base-port "$base"
    load_topology_ports
    if "$SOURCE/python-local" "$SOURCE/scripts/check_dt_ports_free.py" \
        --host 0.0.0.0 --ports "${local_ports[@]}" \
        >"$root/port-probe.log" 2>&1 \
        && probe_remote_ports >"$root/remote-port-probe.log" 2>&1; then
      printf '%s\n' "$base" >"$root/allocated-port-base.txt"
      return 0
    fi
  done
  echo "could not allocate a free dual-host DT port block" >&2
  return 1
}

wait_for_ports_free() {
  local free_polls=0
  local attempt
  for attempt in $(seq 1 180); do
    if "$SOURCE/python-local" "$SOURCE/scripts/check_dt_ports_free.py" \
        --host 0.0.0.0 --ports "${local_ports[@]}" \
        >"$root/port-probe.log" 2>&1 \
        && probe_remote_ports >"$root/remote-port-probe.log" 2>&1; then
      free_polls=$((free_polls + 1))
      if [ "$free_polls" -ge 3 ]; then
        return 0
      fi
    else
      free_polls=0
    fi
    sleep 1
  done
  echo "DT ports did not drain on both hosts within 180 seconds" >&2
  cat "$root/port-probe.log" >&2 || true
  cat "$root/remote-port-probe.log" >&2 || true
  return 1
}

preflight_environment() {
  local required
  for required in \
      "$SOURCE/python-local" \
      "$SOURCE/scripts/dt_target_combo_service.sh" \
      "$SOURCE/scripts/dt_target_combo_benchmark.sh" \
      "$SOURCE/scripts/prepare_dt_combo_topology.py" \
      "$W/profile.toml" \
      "$W/topology-7t2d-page-ab.toml" \
      "$W/prompts.json"; do
    if [ ! -r "$required" ]; then
      echo "preflight: missing or unreadable input: $required" >&2
      return 1
    fi
  done

  "$SOURCE/python-local" - "$W" <<'PY'
import json
import pathlib
import sys
import tomllib

import torch

w = pathlib.Path(sys.argv[1])
with (w / "prompts.json").open("rb") as f:
    prompts = json.load(f)
with (w / "profile.toml").open("rb") as f:
    tomllib.load(f)
with (w / "topology-7t2d-page-ab.toml").open("rb") as f:
    tomllib.load(f)
if not prompts:
    raise SystemExit("preflight: prompts.json is empty")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(
        f"preflight: expected 8 local CUDA devices, got {torch.cuda.device_count()}"
    )
PY

  ssh -o BatchMode=yes -o ConnectTimeout=8 \
      "${DT_SYS4_SSH_HOST:-h8004-lsh-target-ab}" 'bash -s' <<'REMOTE'
set -euo pipefail
test -d /home/lishuhuan/dt-v3-page-run
test -r /share/dai-sys/models/qwen3-14b/config.json
test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -eq 8
REMOTE
}

cleanup_all() {
  cleanup_benchmark
  cleanup_service
}
on_exit() {
  local rc=$?
  cleanup_all
  if [ "$rc" -ne 0 ] && [ -f "$root/status.txt" ]; then
    case "$(cat "$root/status.txt")" in
      waiting|waiting-gpu-idle|preflight|running)
        printf 'failed-runner-rc-%s\n' "$rc" >"$root/status.txt"
        ;;
    esac
  fi
}
trap on_exit EXIT
trap 'exit 130' INT TERM

# Hold one global lease across all arms.  This keeps the comparison contiguous
# and makes nested gpu-lease.sh invocations validation-only via DT_LEASE_HELD=1.
exec 8</data/zhaotianlang/ci/tools/gpu.lock
printf 'waiting\n' >"$root/status.txt"
flock 8
printf 'acquired\n' >"$root/global-lock-acquired"
printf 'preflight\n' >"$root/status.txt"
if ! preflight_environment >"$root/preflight.log" 2>&1; then
  printf 'preflight-failed\n' >"$root/status.txt"
  exit 3
fi
printf 'passed\n' >"$root/preflight-status.txt"
printf 'waiting-gpu-idle\n' >"$root/status.txt"
if ! wait_for_dual_host_gpu_idle >"$root/gpu-idle-wait.log" 2>&1; then
  printf 'gpu-idle-wait-failed\n' >"$root/status.txt"
  exit 3
fi
printf 'running\n' >"$root/status.txt"
if ! allocate_topology; then
  printf 'port-allocation-failed\n' >"$root/status.txt"
  exit 3
fi
export DT_COMBO_TOPOLOGY_FILE="$topology"
wait_for_ports_free

for label in "${arms[@]}"; do
  arm_root="$root/$label"
  mkdir -p "$arm_root"
  printf 'starting\n' >"$arm_root/status.txt"

  setsid "$SOURCE/scripts/dt_target_combo_service.sh" "$stamp" "$label" \
    >"$arm_root/service.log" 2>&1 &
  service_pid=$!
  printf '%s\n' "$service_pid" >"$arm_root/service.pid"

  "$SOURCE/scripts/dt_target_combo_benchmark.sh" "$stamp" "$label" \
    >"$arm_root/benchmark.log" 2>&1 &
  benchmark_pid=$!
  while kill -0 "$benchmark_pid" 2>/dev/null; do
    if ! kill -0 "$service_pid" 2>/dev/null; then
      echo "service exited while benchmark was waiting/running: $label" \
        >>"$arm_root/benchmark.log"
      cleanup_benchmark
      break
    fi
    sleep 1
  done
  benchmark_rc=0
  if [ -n "${benchmark_pid:-}" ]; then
    wait "$benchmark_pid" || benchmark_rc=$?
    benchmark_pid=
  else
    benchmark_rc=5
  fi
  if [ "$benchmark_rc" -ne 0 ]; then
    printf 'benchmark-failed\n' >"$arm_root/status.txt"
    printf 'failed-%s\n' "$label" >"$root/status.txt"
    cleanup_service
    wait_for_ports_free || true
    exit 4
  fi

  cleanup_service
  wait_for_ports_free
  printf 'complete\n' >"$arm_root/status.txt"
done

trap - EXIT INT TERM
"$SOURCE/python-local" "$SOURCE/scripts/summarize_dt_target_combo.py" \
  "$root" --output "$root/summary.csv" >"$root/summary.md"
printf 'complete\n' >"$root/status.txt"
echo "$root"
