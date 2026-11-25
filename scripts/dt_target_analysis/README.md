# DT V3 Target optimization analysis

This directory contains the portable analysis utilities used for the Target-side
experiments summarized in `docs/dt_target_optimization_attempts.md`.

The scripts consume exported DT profile directories or benchmark JSON/JSONL
files. They do not launch services, reserve GPUs, or contain cluster addresses.
Raw traces and model outputs are intentionally not committed.

Main entry points:

- `slo_interp.py`: interpolate the maximum concurrency/throughput satisfying
  TPOT p90 <= 30 ms.
- `tpot.py`: summarize request TPOT distributions.
- `hostpath.py`, `loop_iter.py`, `loop_tail.py`: attribute Target scheduler
  host-path and launch-iteration latency.
- `ready_join.py`, `observe_lag.py`, `cand_freeze.py`: join candidate publish,
  observation, READY and freeze events.
- `draft.py`, `draft_balance.py`, `draft_gaps.py`: quantify the Draft-side
  contribution needed to interpret Target changes.
- `bench_verify_shapes.py`, `pick_buckets.py`: inspect verify CUDA-graph bucket
  shapes.
- `eval_quality.py`: compare BF16/FP8 quality runs.

All behavior-changing code paths in this PR remain opt-in through environment
flags. See the summary document for the exact flag, measurement scope and
result of each attempt.
