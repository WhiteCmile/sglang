# Speculative Decoding Call Chain

This document summarizes the main call chain of SGLang speculative decoding for the `spec v2` / `EAGLE` path, starting from prefill and then entering speculative decode iterations.

The focus here is on:

- which top-level function is called at each stage
- how data moves between `ScheduleBatch`, `ModelWorkerBatch`, and `ForwardBatch`
- where draft, verify, and draft-extend happen

## Core Data Structures

The main batch abstraction changes across three layers:

```text
ScheduleBatch -> ModelWorkerBatch -> ForwardBatch
```

Reference:

- [forward_batch_info.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/model_executor/forward_batch_info.py#L17)

## Scheduler Entry

The scheduler launches one forward from:

- [scheduler.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/managers/scheduler.py#L2484)

```python
batch_result = self.model_worker.forward_batch_generation(model_worker_batch)
```

When speculative decoding is enabled, `self.model_worker` is the draft/spec worker rather than the normal `tp_worker`.

Reference:

- [scheduler.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/managers/scheduler.py#L2524)

## Prefill Phase

### 1. Scheduler calls the speculative worker

Prefill enters:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L695)

At this point the batch is still in extend/prefill mode, so the worker first runs the target model.

### 2. Target worker runs the prefill forward

The target worker call is:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L695)

```python
batch_output = self.target_worker.forward_batch_generation(model_worker_batch)
```

This then enters:

- [tp_worker.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/managers/tp_worker.py#L447)

Inside `TpModelWorker.forward_batch_generation(...)`:

1. `ModelWorkerBatch` is converted to `ForwardBatch`
2. `ModelRunner.forward(...)` is called

References:

- [tp_worker.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/managers/tp_worker.py#L463)
- [tp_worker.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/managers/tp_worker.py#L471)

### 3. ModelRunner executes the actual prefill

The actual model execution enters:

- [model_runner.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/model_executor/model_runner.py#L2461)

```python
output = self._forward_raw(forward_batch, ...)
```

So the prefill path is:

```text
Scheduler
  -> EAGLEWorkerV2.forward_batch_generation
    -> target_worker.forward_batch_generation
      -> ForwardBatch.init_new
      -> ModelRunner.forward
        -> ModelRunner._forward_raw
```

### 4. Draft side prepares the first speculative state

After target prefill finishes, the draft worker does one extra step to prepare the starting draft state for speculative decode:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L701)

```python
batch_output.next_draft_input = self.draft_worker._draft_extend_for_prefill(
    model_worker_batch,
    batch_output.logits_output.hidden_states,
    batch_output.next_token_ids,
    ...
)
```

This function uses:

- target hidden states
- target next token ids

to produce the first `next_draft_input`.

Reference:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L448)

## One Speculative Decode Iteration

Once the batch is no longer in prefill, each speculative decode iteration has three stages:

```text
draft -> verify -> draft_extend
```

The controlling function is:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L715)

## Draft Stage

### 1. Draft worker runs drafting

The decode path starts with:

```python
verify_input = self.draft_worker.draft(model_worker_batch)
```

Reference:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L727)

Inside `draft(...)`, the worker:

1. prepares draft inputs with `prepare_for_v2_draft(...)`
2. runs the draft model
3. builds the verify tree and mask

Reference:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L320)

### 2. Verify chain is built by prepending `verified_id`

The most important tree-building step is:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L361)
- [eagle_utils.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_utils.py#L47)

```python
build_tree_kernel_efficient(
    draft_input.verified_id,
    ...,
    draft_tokens,
    ...
)
```

Inside `build_tree_kernel_efficient(...)`:

```python
draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()
```

For `topk=1`, if:

- current verified token is `X`
- draft predicts `Y Z W`

then the verify chain is effectively:

```text
X Y Z W
```

## Verify Stage

### 1. The draft output becomes `batch.spec_info`

After `draft(...)`, the result is an `EagleVerifyInput`:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L731)

```python
model_worker_batch.spec_info = verify_input
batch_output = self.verify(model_worker_batch)
```

### 2. Prepare target verify forward

Inside `verify(...)`, the target verify `ForwardBatch` is prepared here:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L756)

```python
verify_forward_batch, can_run_cuda_graph = verify_input.prepare_for_v2_verify(
    self.req_to_token_pool,
    batch,
    self.target_worker,
)
```

### 3. Target model executes verify

The actual target verify forward is:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L791)

```python
forward_batch_output = self.target_worker.forward_batch_generation(
    model_worker_batch=None,
    forward_batch=verify_forward_batch,
    is_verify=True,
    skip_attn_backend_init=True,
)
```

Since `is_verify=True`, `TpModelWorker.forward_batch_generation(...)` returns logits directly and skips normal sampling.

Reference:

- [tp_worker.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/managers/tp_worker.py#L484)

### 4. Accept/reject decisions are computed

After target logits are produced:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L820)

```python
predict, accept_length, accept_index = verify_input.sample(...)
```

Then the new verified token is extracted:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L841)

```python
all_verified_id = predict[accept_index]
fill_new_verified_id[(bs,)](
    all_verified_id,
    accept_length,
    verified_id,
    self.speculative_num_draft_tokens,
)
```

### 5. The next round's draft input is created

At the end of `verify(...)`:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L856)

```python
next_draft_input = EagleDraftInput(
    verified_id=verified_id,
    new_seq_lens=new_seq_lens,
    verify_done=verify_done,
)
```

This is the skeleton of the next round's draft input.

At this point it contains:

- the new `verified_id`
- the updated sequence lengths
- an event marking verify completion

but not yet the final `topk_p`, `topk_index`, or `hidden_states`.

## Draft Extend Stage

After verify, the draft worker still needs to align its own state with the verified target state.

That happens here:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L733)

```python
self.draft_worker._draft_extend_for_decode(model_worker_batch, batch_output)
```

Inside `_draft_extend_for_decode(...)`:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L547)

it:

1. constructs a draft-extend batch
2. runs the draft model again
3. picks the state at the verified continuation point
4. fills `next_draft_input.topk_p/topk_index/hidden_states`

The final write-back is:

- [eagle_worker_v2.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_worker_v2.py#L600)

```python
next_draft_input = batch_result.next_draft_input
next_draft_input.topk_p = ret_topk_p
next_draft_input.topk_index = ret_topk_index
next_draft_input.hidden_states = ret_hidden_states
```

After this step, `next_draft_input` is fully ready for the next speculative decode iteration.

## Full Call Chain Summary

### Prefill

```text
Scheduler
  -> EAGLEWorkerV2.forward_batch_generation
    -> target_worker.forward_batch_generation
      -> ForwardBatch.init_new
      -> ModelRunner.forward
        -> ModelRunner._forward_raw
    -> draft_worker._draft_extend_for_prefill
      -> draft_runner.forward
      -> produce next_draft_input
```

### One speculative decode iteration

```text
Scheduler
  -> EAGLEWorkerV2.forward_batch_generation
    -> draft_worker.draft
      -> prepare_for_v2_draft
      -> draft_runner.forward
      -> build_tree_kernel_efficient
    -> verify
      -> prepare_for_v2_verify
      -> target_worker.forward_batch_generation(... is_verify=True)
        -> ModelRunner.forward
      -> sample / fill_new_verified_id
      -> construct next_draft_input
    -> draft_worker._draft_extend_for_decode
      -> draft_runner.forward
      -> fill next_draft_input.topk_p/topk_index/hidden_states
```

### Important invariant

For `topk=1`, each speculative iteration verifies a chain of the form:

```text
[verified_id, draft_token_1, draft_token_2, draft_token_3]
```

That leading `verified_id` is inserted by:

- [eagle_utils.py](/Users/zhaotianlang/Local-Codes/sglang/python/sglang/srt/speculative/eagle_utils.py#L61)

So if:

- current verified token is `X`
- draft predicts `Y Z W`

then target verify operates on the logical chain:

```text
X Y Z W
```
