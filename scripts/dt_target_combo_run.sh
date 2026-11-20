#!/bin/bash
set -euo pipefail

W=${DT_EXPERIMENT_WORKSPACE:-/data/lishuhuan/dt-v3-pilot}
SOURCE=${DT_EXPERIMENT_SOURCE:-/data/lishuhuan/dt-v3-page}
stamp=${1:-$(date +%Y%m%d-%H%M%S)-target-combo}
root="$W/target-combo-results/$stamp"
arms=(control early_slots8 early_page64 early_page64_slots8)

mkdir -p "$root"
printf '%s\n' "$stamp" >"$root/stamp.txt"
git -C "$SOURCE" rev-parse HEAD >"$root/source-commit.txt"
printf '%s\n' "${arms[@]}" >"$root/arms.txt"

service_pid=
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
trap cleanup_service EXIT INT TERM

# Hold one global lease across all arms.  This keeps the comparison contiguous
# and makes nested gpu-lease.sh invocations validation-only via DT_LEASE_HELD=1.
exec 8</data/zhaotianlang/ci/tools/gpu.lock
printf 'waiting\n' >"$root/status.txt"
flock 8
printf 'acquired\n' >"$root/global-lock-acquired"
printf 'running\n' >"$root/status.txt"

for label in "${arms[@]}"; do
  arm_root="$root/$label"
  mkdir -p "$arm_root"
  printf 'starting\n' >"$arm_root/status.txt"

  setsid "$SOURCE/scripts/dt_target_combo_service.sh" "$stamp" "$label" \
    >"$arm_root/service.log" 2>&1 &
  service_pid=$!
  printf '%s\n' "$service_pid" >"$arm_root/service.pid"

  if ! "$SOURCE/scripts/dt_target_combo_benchmark.sh" "$stamp" "$label" \
      >"$arm_root/benchmark.log" 2>&1; then
    printf 'benchmark-failed\n' >"$arm_root/status.txt"
    cleanup_service
    exit 4
  fi

  cleanup_service
  printf 'complete\n' >"$arm_root/status.txt"
done

trap - EXIT INT TERM
"$SOURCE/python-local" "$SOURCE/scripts/summarize_dt_target_combo.py" \
  "$root" --output "$root/summary.csv" >"$root/summary.md"
printf 'complete\n' >"$root/status.txt"
echo "$root"
