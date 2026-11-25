#!/bin/bash
set -euo pipefail

W=${DT_EXPERIMENT_WORKSPACE:-/data/lishuhuan/dt-v3-pilot}
SOURCE=${DT_EXPERIMENT_SOURCE:-/data/lishuhuan/dt-v3-page}
stamp=${1:?stamp required}
label=${2:?label required}
root="$W/target-combo-results/$stamp/$label"
topology=${DT_COMBO_TOPOLOGY_FILE:-$W/topology-7t2d-page-ab.toml}

mkdir -p "$root"

# Every arm starts from the same Target policy.  Only the three allocator/result
# knobs named by the arm are allowed to differ.  The 2-lane code default is four
# physical source slots; spell it out so a later default change cannot skew A/B.
unset DT_PROFILE_DIR DT_PROFILE_EVENTS
unset DT_TARGET_READY_REFRESH_BEFORE_SELECT DT_TARGET_ADAPTIVE_READY_HOLD
unset DT_TARGET_EARLY_READY_RESULT DT_TARGET_LATE_FREEZE
unset DT_TARGET_BATCH_WAIT_MS DT_FEATURE_SOURCE_SLOTS
unset DT_FEATURE_SOURCE_PAGE_ROWS

case "$label" in
  control)
    export DT_FEATURE_SOURCE_SLOTS=4
    ;;
  early_slots8)
    export DT_TARGET_EARLY_READY_RESULT=1
    export DT_FEATURE_SOURCE_SLOTS=8
    ;;
  early_page64)
    export DT_TARGET_EARLY_READY_RESULT=1
    export DT_FEATURE_SOURCE_PAGE_ROWS=64
    export DT_FEATURE_SOURCE_SLOTS=4
    ;;
  early_page64_slots8)
    export DT_TARGET_EARLY_READY_RESULT=1
    export DT_FEATURE_SOURCE_PAGE_ROWS=64
    export DT_FEATURE_SOURCE_SLOTS=8
    ;;
  *)
    echo "unknown combo arm: $label" >&2
    exit 2
    ;;
esac

# Draft is deliberately fixed: this experiment only changes Target behavior.
export DT_DRAFT_FEATURE_BUFFER_SLOT_COUNT=2
export DT_LEASE_DIR="$W/lease"
export DT_LEASE_HELD=1

cat >"$root/effective-flags.txt" <<EOF
DT_TARGET_EARLY_READY_RESULT=${DT_TARGET_EARLY_READY_RESULT:-0}
DT_FEATURE_SOURCE_PAGE_ROWS=${DT_FEATURE_SOURCE_PAGE_ROWS:-0}
DT_FEATURE_SOURCE_SLOTS=$DT_FEATURE_SOURCE_SLOTS
DT_DRAFT_FEATURE_BUFFER_SLOT_COUNT=$DT_DRAFT_FEATURE_BUFFER_SLOT_COUNT
DT_TARGET_LATE_FREEZE=default
DT_TARGET_BATCH_WAIT_MS=default
EOF

exec timeout --signal=INT --kill-after=120s 7200s \
  "$W/ci/gpu-lease.sh" "$SOURCE/python-local" -m dt_launcher dt \
  --workspace "$W" --dt-sglang-dir "$SOURCE" --profile "$W/profile.toml" \
  --preset generic --topology-file "$topology" \
  --python-executable "$SOURCE/python-local" --candidate-transport cuda-ipc \
  --run-dir "$root/run-$$" \
  --max-running-requests 384 --max-pending-tasks 512 \
  --max-draft-batch-size 128 --target-mem-fraction-static 0.70 \
  --draft-mem-fraction-static 0.75 --target-overlap-schedule \
  --target-overlap-result-defer-ms 32 --target-round-timeout-ms 120000
