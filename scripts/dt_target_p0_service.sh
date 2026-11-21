#!/bin/bash
set -euo pipefail

W=${DT_EXPERIMENT_WORKSPACE:-/data/lishuhuan/dt-v3-pilot}
SOURCE=${DT_EXPERIMENT_SOURCE:-/data/lishuhuan/dt-v3-page}
stamp=${1:?stamp required}
label=${2:-page64-p0}
profile_dir="/tmp/dt-issue281-p0-$stamp"
root="$W/issue281-p0-results/$stamp/$label"
local_ports=(37900 37901 37902 37920 37921 38100 38101 38102 51121 51122)

unset DT_TARGET_READY_REFRESH_BEFORE_SELECT DT_TARGET_ADAPTIVE_READY_HOLD
unset DT_TARGET_EARLY_READY_RESULT DT_TARGET_LATE_FREEZE
unset DT_TARGET_BATCH_WAIT_MS

export DT_FEATURE_SOURCE_PAGE_ROWS=64
export DT_FEATURE_SOURCE_SLOTS=4
export DT_DRAFT_FEATURE_BUFFER_SLOT_COUNT=2
export DT_PROFILE_DIR="$profile_dir"
export DT_PROFILE_EVENTS=target_p0_hop
export DT_GPU_CLOCK=1
export DT_LEASE_DIR="$W/lease"

mkdir -p "$root"
exec 8</data/zhaotianlang/ci/tools/gpu.lock
flock 8

# A previous launcher's process group can outlive the process that held the
# lease for a few seconds.  Do not hand that teardown race to dt_launcher.
free_polls=0
for _ in $(seq 1 180); do
  if ! "$SOURCE/python-local" "$SOURCE/scripts/check_dt_ports_free.py" \
      --host 192.168.1.249 --ports "${local_ports[@]}" \
      >"$root/port-probe.log" 2>&1; then
    free_polls=0
  else
    free_polls=$((free_polls + 1))
    [ "$free_polls" -ge 3 ] && break
  fi
  sleep 1
done
if [ "$free_polls" -lt 3 ]; then
  echo "local DT ports did not drain before P0 launch" >&2
  cat "$root/port-probe.log" >&2 || true
  exit 3
fi

printf 'acquired\n' >"$root/global-lock-acquired"
export DT_LEASE_HELD=1
exec timeout --signal=INT --kill-after=120s 7200s \
  "$W/ci/gpu-lease.sh" "$SOURCE/python-local" -m dt_launcher dt \
  --workspace "$W" --dt-sglang-dir "$SOURCE" --profile "$W/profile.toml" \
  --preset generic --topology-file "$W/topology-7t2d-page-ab.toml" \
  --python-executable "$SOURCE/python-local" --candidate-transport cuda-ipc \
  --run-dir "$root/run-$$" \
  --max-running-requests 384 --max-pending-tasks 512 \
  --max-draft-batch-size 128 --target-mem-fraction-static 0.70 \
  --draft-mem-fraction-static 0.75 --target-overlap-schedule \
  --target-overlap-result-defer-ms 32 --target-round-timeout-ms 120000
