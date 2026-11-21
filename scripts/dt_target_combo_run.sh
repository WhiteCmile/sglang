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
        --host 192.168.1.249 --ports "${local_ports[@]}" \
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
