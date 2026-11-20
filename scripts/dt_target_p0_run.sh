#!/bin/bash
set -euo pipefail

W=${DT_EXPERIMENT_WORKSPACE:-/data/lishuhuan/dt-v3-pilot}
SOURCE=${DT_EXPERIMENT_SOURCE:-/data/lishuhuan/dt-v3-page}
stamp=${1:-$(date +%Y%m%d-%H%M%S)-issue281-p0}
label=page64-p0
root="$W/issue281-p0-results/$stamp"
profile_dir="/tmp/dt-issue281-p0-$stamp"
local_ports_re=':(37900|37901|37902|37920|37921|38100|38101|38102|51121|51122)\b'

mkdir -p "$root/$label"
printf '%s\n' "$stamp" >"$root/stamp.txt"
git -C "$SOURCE" rev-parse HEAD >"$root/source-commit.txt"
printf 'waiting\n' >"$root/status.txt"

service_pid=
benchmark_pid=
cleanup() {
  if [ -n "${benchmark_pid:-}" ] && kill -0 "$benchmark_pid" 2>/dev/null; then
    kill -TERM "$benchmark_pid" 2>/dev/null || true
  fi
  if [ -n "${benchmark_pid:-}" ]; then
    wait "$benchmark_pid" 2>/dev/null || true
  fi
  benchmark_pid=
  if [ -n "${service_pid:-}" ] && kill -0 "$service_pid" 2>/dev/null; then
    kill -INT -- "-$service_pid" 2>/dev/null || true
    for _ in $(seq 1 90); do
      kill -0 "$service_pid" 2>/dev/null || break
      sleep 2
    done
    kill -0 "$service_pid" 2>/dev/null && \
      kill -TERM -- "-$service_pid" 2>/dev/null || true
  fi
  if [ -n "${service_pid:-}" ]; then
    wait "$service_pid" 2>/dev/null || true
  fi
  service_pid=
}
trap cleanup EXIT INT TERM

setsid "$SOURCE/scripts/dt_target_p0_service.sh" "$stamp" "$label" \
  >"$root/$label/service.log" 2>&1 &
service_pid=$!
printf '%s\n' "$service_pid" >"$root/$label/service.pid"

while [ ! -f "$root/$label/global-lock-acquired" ]; do
  if ! kill -0 "$service_pid" 2>/dev/null; then
    echo "P0 service exited before acquiring the GPU lock" >&2
    exit 4
  fi
  sleep 2
done
printf 'running\n' >"$root/status.txt"

"$SOURCE/scripts/dt_target_p0_benchmark.sh" "$stamp" "$label" \
  >"$root/$label/benchmark.log" 2>&1 &
benchmark_pid=$!
while kill -0 "$benchmark_pid" 2>/dev/null; do
  if ! kill -0 "$service_pid" 2>/dev/null; then
    echo "P0 service exited while benchmark was waiting/running" \
      >>"$root/$label/benchmark.log"
    kill -TERM "$benchmark_pid" 2>/dev/null || true
    wait "$benchmark_pid" 2>/dev/null || true
    benchmark_pid=
    printf 'service-failed\n' >"$root/status.txt"
    exit 5
  fi
  sleep 1
done
benchmark_rc=0
wait "$benchmark_pid" || benchmark_rc=$?
benchmark_pid=
if [ "$benchmark_rc" -ne 0 ]; then
  printf 'benchmark-failed\n' >"$root/status.txt"
  exit "$benchmark_rc"
fi

cleanup
for _ in $(seq 1 180); do
  ss -H -ltn | grep -Eq "$local_ports_re" || break
  sleep 1
done
trap - EXIT INT TERM

mkdir -p "$root/profiles/sys3" "$root/profiles/sys4" "$root/analysis"
if [ -d "$profile_dir" ]; then
  cp -a "$profile_dir"/. "$root/profiles/sys3"/
fi
rsync -a h8004-lsh-target-ab:"$profile_dir/" "$root/profiles/sys4/"
"$SOURCE/python-local" "$SOURCE/scripts/analyze_dt_target_p0.py" \
  "$root/profiles" --output-dir "$root/analysis" \
  --request-prefix dt-benchmark-measurement- \
  >"$root/analysis/summary.stdout.json"

printf 'complete\n' >"$root/status.txt"
echo "$root"
