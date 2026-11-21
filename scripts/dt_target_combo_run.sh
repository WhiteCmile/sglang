#!/bin/bash
set -euo pipefail

W=${DT_EXPERIMENT_WORKSPACE:-/data/lishuhuan/dt-v3-pilot}
SOURCE=${DT_EXPERIMENT_SOURCE:-/data/lishuhuan/dt-v3-page}
stamp=${1:-$(date +%Y%m%d-%H%M%S)-target-combo}
root="$W/target-combo-results/$stamp"
arms=(control early_slots8 early_page64 early_page64_slots8)
local_ports=(37900 37901 37902 37920 37921 38100 38101 38102 51121 51122)

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

wait_for_local_ports_free() {
  local free_polls=0
  local attempt
  for attempt in $(seq 1 180); do
    if ! "$SOURCE/python-local" "$SOURCE/scripts/check_dt_ports_free.py" \
        --host 0.0.0.0 --ports "${local_ports[@]}" \
        >"$root/port-probe.log" 2>&1; then
      free_polls=0
    else
      free_polls=$((free_polls + 1))
      if [ "$free_polls" -ge 3 ]; then
        return 0
      fi
    fi
    sleep 1
  done
  echo "local DT ports did not drain within 180 seconds" >&2
  cat "$root/port-probe.log" >&2 || true
  return 1
}

preflight_environment() {
  local required
  for required in \
      "$SOURCE/python-local" \
      "$SOURCE/scripts/dt_target_combo_service.sh" \
      "$SOURCE/scripts/dt_target_combo_benchmark.sh" \
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

  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo "preflight: sys-3 still has active GPU compute processes" >&2
    return 1
  fi

  ssh -o BatchMode=yes -o ConnectTimeout=8 \
      "${DT_SYS4_SSH_HOST:-h8004-lsh-target-ab}" 'bash -s' <<'REMOTE'
set -euo pipefail
test -d /home/lishuhuan/dt-v3-page-run
test -r /share/dai-sys/models/qwen3-14b/config.json
test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -eq 8
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "preflight: sys-4 still has active GPU compute processes" >&2
  exit 1
fi
python3 - <<'PY'
import socket

ports = [
    37900, 37901, 37902, 37903,
    38100, 38101, 38102, 38103,
    41300, 41301, 41302, 41303,
]
busy = []
for port in ports:
    sock = socket.socket()
    try:
        sock.bind(("0.0.0.0", port))
    except OSError as exc:
        busy.append((port, str(exc)))
    finally:
        sock.close()
if busy:
    raise SystemExit(f"preflight: sys-4 ports unavailable: {busy}")
PY
REMOTE
}

cleanup_all() {
  cleanup_benchmark
  cleanup_service
}
trap cleanup_all EXIT INT TERM

# Hold one global lease across all arms.  This keeps the comparison contiguous
# and makes nested gpu-lease.sh invocations validation-only via DT_LEASE_HELD=1.
exec 8</data/zhaotianlang/ci/tools/gpu.lock
printf 'waiting\n' >"$root/status.txt"
flock 8
printf 'acquired\n' >"$root/global-lock-acquired"
printf 'running\n' >"$root/status.txt"
wait_for_local_ports_free
if ! preflight_environment >"$root/preflight.log" 2>&1; then
  printf 'preflight-failed\n' >"$root/status.txt"
  exit 3
fi
printf 'passed\n' >"$root/preflight-status.txt"

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
    cleanup_service
    wait_for_local_ports_free || true
    exit 4
  fi

  cleanup_service
  wait_for_local_ports_free
  printf 'complete\n' >"$arm_root/status.txt"
done

trap - EXIT INT TERM
"$SOURCE/python-local" "$SOURCE/scripts/summarize_dt_target_combo.py" \
  "$root" --output "$root/summary.csv" >"$root/summary.md"
printf 'complete\n' >"$root/status.txt"
echo "$root"
