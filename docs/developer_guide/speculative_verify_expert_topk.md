# Speculative Verify Expert Top-K Capture

This note records the recent changes for capturing MoE expert routing during speculative decoding verification (EAGLE/NEXTN path).

## Goal

Capture and persist target-model expert `topk_ids` for each speculative **target verify** forward, while keeping similarity analysis and plotting fully offline.

## What Was Added

### 1. New server argument

- `--speculative-verify-expert-topk-output-dir <dir>`

When set:

- capture is enabled for speculative target verify forwards;
- recorder mode is auto-set to `per_token` if `expert_distribution_recorder_mode` is unset;
- `expert_distribution_recorder_mode` must be `per_token` or `per_pass` if explicitly set.

## 2. Recorder object output support

The expert distribution detail accumulator now supports:

- `dump_record(output_mode="object")`

This enables in-process access to per-forward captured data instead of only writing recorder files.

## 3. Target verify dump in EAGLE worker

During `EAGLEWorker.verify()`:

- a strict recorder window is opened only for this verify call (`stop -> start`);
- the target forward is recorded;
- recorder output is immediately dumped to a `.pt` file;
- recorder is always stopped in `finally`, so non-verify forwards are excluded;
- verify tree metadata is saved together with expert records.

Output filename pattern:

- `verify_expert_topk_tp{rank}_{index}.pt`

Examples:

- `verify_expert_topk_tp0_00000000.pt`
- `verify_expert_topk_tp7_00000042.pt`

## Dumped Payload Structure

Each file contains:

- `record_index`: incremental local index
- `bid`: model worker batch id
- `forward_pass_id`: recorder forward id
- `records`: recorder records containing per-layer `topk_ids_of_layer`
- `verify_tree_meta`:
  - `draft_token`
  - `positions`
  - `retrive_index`
  - `retrive_next_token`
  - `retrive_next_sibling`
  - `parent_index`
  - `draft_token_num`
  - `spec_steps`
  - `topk`

## Helper Scripts

### Launch script

- `scripts/playground/run_deepseek_v3_verify_expert_topk.sh`

Example:

```bash
DEEPSEEK_PATH=/path/to/deepseek-v3 \
EXPERT_OUT_DIR=/tmp/deepseek_verify_expert_topk \
bash scripts/playground/run_deepseek_v3_verify_expert_topk.sh
```

### Offline reader skeleton

- `scripts/playground/read_verify_expert_topk.py`

Example:

```bash
python3 scripts/playground/read_verify_expert_topk.py \
  --input-dir /tmp/deepseek_verify_expert_topk \
  --max-files 20
```

### Offline similarity analysis by tree depth

- `scripts/playground/analyze_verify_expert_topk_similarity.py`

Compute expert-routing similarity for each `(moe_layer, request, tree_depth, parent_slot)` group:

- primary metric: `intersection_size / experts_per_token`  
  where `intersection_size` is the intersection across all child tokens under one parent;
- reference metric: `intersection_size / union_size`.

The script groups by `(request, depth, parent_slot)` first, then aggregates to `(file, layer)`
with weighted averages:

- `weighted_inter_over_token_expert_count = sum(intersection_size) / sum(experts_per_token)`
- `weighted_inter_over_union = sum(intersection_size) / sum(union_size)`

Example:

```bash
python3 scripts/playground/analyze_verify_expert_topk_similarity.py \
  --input-dir /tmp/deepseek_verify_expert_topk \
  --max-files 200 \
  --min-tokens-per-group 2 \
  --output-csv /tmp/verify_expert_similarity.csv
```

CSV output behavior:

- default (`--csv-final-only`): write final **request-level** averages per `.pt` file;
- `--csv-include-groups`: write per-group rows (`request, depth, parent_slot`) instead.

### Real-data feeder script

- `scripts/playground/feed_real_data_for_verify.py`

Feed local JSON/JSONL prompts:

```bash
python3 scripts/playground/feed_real_data_for_verify.py \
  --base-url http://127.0.0.1:30001/v1 \
  --dataset-source local \
  --dataset-path /path/to/real_dataset.jsonl \
  --num-batches 100 \
  --batch-size 8 \
  --max-concurrency 8 \
  --max-tokens 128
```

Feed from HuggingFace UltraChat:

```bash
python3 scripts/playground/feed_real_data_for_verify.py \
  --base-url http://127.0.0.1:30001/v1 \
  --dataset-source ultrachat \
  --num-batches 100 \
  --batch-size 8 \
  --max-concurrency 8 \
  --max-tokens 128
```

## Multi-Batch Semantics

- One `.pt` file corresponds to one **target verify forward pass**.
- Under concurrent traffic, a verify forward may contain requests from multiple client-side batches.
- Use `bid`, `forward_pass_id`, and `verify_tree_meta` inside each file for offline alignment.

## Verify-Only Capture Semantics

- Recorder lifetime is scoped to each `EAGLEWorker.verify()` call.
- Non-verify passes (for example prefill/decode between verify calls) are not expected to be captured.
- This avoids mixed windows where `records` could include unrelated forwards.

## Scope Notes

- This change only captures and saves routing data.
- No in-engine similarity computation is added.
- Intended workflow: capture online, analyze offline.

## Hotfix: Dynamic Top-K Width

### Background

In some DeepSeek runs, `topk_ids.shape[1]` can be larger than 8 (for example 9 in certain routing configurations).  
The original per-token recorder implementation used a fixed top-k buffer width of 8, which could cause shape mismatch errors during assignment.

### Fix

The per-token detail gatherer now:

- starts from a default top-k buffer width;
- dynamically expands the last dimension when a larger `topk_ids.shape[1]` is observed;
- tracks the max observed top-k width in the current pass;
- trims output tensors to the observed width at `collect()` time.

This keeps backward compatibility while preventing runtime crashes caused by fixed-width assumptions.
