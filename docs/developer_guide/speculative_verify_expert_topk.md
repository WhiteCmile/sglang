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

- the target forward is recorded;
- recorder output is immediately dumped to a `.pt` file;
- verify tree metadata is saved together with expert records.

Output filename pattern:

- `verify_expert_topk_00000000.pt`

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

## Scope Notes

- This change only captures and saves routing data.
- No in-engine similarity computation is added.
- Intended workflow: capture online, analyze offline.
