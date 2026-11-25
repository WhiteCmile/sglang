# Target feature-source buffer: classic designs and a measurement-gated choice

Status: design document for issue #281; proposes work only. It does not change
Target, Draft, Coordinator, test, or production code.

Safety boundary: Target-side scope only. Draft and Coordinator changes below are
future protocol work requiring owner approval. Every design must preserve
source-slot generation, read-proof-before-reuse, TP lockstep, request/KV
ownership, and exact-token semantics.

## 1. What the current Target source buffer is

`TargetFeatureSourcePool` (`python/sglang/srt/speculative/dt_feature_p2p.py:208`)
owns one contiguous BF16 allocation shaped
`[physical_slot_count, max_tokens, feature_width]`, registered once
(`dt_feature_p2p.py:279-291`). A publication is one logical READY identity
(`source_slot_id`, `slot_generation`) plus a contiguous row range.
`FeatureSourceDescriptor` has one `byte_offset` and one `nbytes` per transfer
(`dt_feature_p2p.py:85-141`); there is no page-list field.

When `DT_FEATURE_SOURCE_PAGE_ROWS` is set, `TargetSourcePageAllocator`
(`python/sglang/srt/speculative/dt_source_page_allocator.py:57`) splits every
physical slot into fixed-size pages. `SourceExtentLease` is still one
`(slot_id, page_start, page_count)` extent (`dt_source_page_allocator.py:25-47`):
`allocate()` best-fits one contiguous extent inside one physical slot and
rejects a request larger than `slot_rows` (`dt_source_page_allocator.py:81-134`);
`release()` coalesces adjacent extents in that slot
(`dt_source_page_allocator.py:136-164`). A physical slot's generation advances
only when its lease set becomes empty (`dt_source_page_allocator.py:113-117`).

The pool maps that extent into the real address space in
`_reserve_slot_locked` (`dt_feature_p2p.py:367-412`):

- logical `slot_count` becomes `min(total_page_count, 4096/4)` so the remote
  generation table still fits one 4 KiB page (`dt_feature_p2p.py:263-271`);
- `allocation_row_offset = extent.slot_id * max_tokens + extent.row_offset`;
- `publication_offset()` returns one base byte offset
  (`dt_feature_p2p.py:326-338`).

A publication sets `slot.active_transfers` to all of its transfers
(`dt_feature_p2p.py:489-497`). `read_done()` removes one transfer, but
`_free_slot_locked()` and therefore the page lease are released only when the
last transfer acks (`dt_feature_p2p.py:719-756`, `:414-432`). The lease is thus
all-or-nothing across a publication.

This is **contiguous suballocation, not true paging**. There is no page table, no
virtual-to-physical page list, no per-page refcount, and no descriptor or wire
record that can name more than one run. "Page64" only changes allocation
granularity from one whole physical slot to a 64-row extent inside that slot.
It reduces internal waste; it does not decouple logical identity from physical
placement beyond picking one contiguous extent.

Current consumers still assume one contiguous run: same-host inline copy is one
`cudaMemcpyAsync` per transfer range (`DraftFeaturePullTransport`,
`CudaIpcRuntime.copy_batch_async`); ring v2 carries one
`source_byte_offset`/`nbytes`/`row_count` and the native worker enqueues one
pull (`csrc/dt_ring_worker.cpp:464-516`); the remote worker maps one record to
one READ (`serve_remote_record`); Draft validates
`source.allocation_offset + source.byte_offset == entry.source_offset` and one
`nbytes` per task (`python/sglang/srt/speculative/dt_draft_engine.py:2640-2660`).

One opening matters for Target-only work: ring v2 validates
`source_byte_offset` against the **whole registered allocation**, not against
`slot_nbytes` (`dt_ring_worker.cpp:468-484`). The comment there explicitly calls
`source_slot_id` a logical READY identity and `source_byte_offset` authoritative.
The inline CUDA-IPC path also validates against `allocation_length`
(`dt_draft_engine.py:10072-10079`). Therefore a *contiguous* extent that crosses
an old physical-slot boundary is representable today; a *non-contiguous* list
of runs is not.

### P0 observation is not a bottleneck finding

The P0 arm sets `DT_FEATURE_SOURCE_PAGE_ROWS=64` and
`DT_FEATURE_SOURCE_SLOTS=4` (`scripts/dt_target_p0_service.sh:16-17`), and
`scripts/analyze_dt_target_p0.py:251-270` computes the page high-water as
`source_pages_total - source_pages_free` from `target_p0_hop` rows. The issue
reports a sampled high-water of three pages. Total pages are
`4 * (max_tokens / 64)`, so three active pages is a tiny fraction of page
capacity. That is evidence **against page-capacity exhaustion** for that
workload, not evidence that page64 is or is not the bottleneck. No allocator
conclusion is valid until M1-M4 distinguish page capacity, logical identity
pressure, pin lifetime, and control-path delay.

## 2. Four distinct pressures

1. **Byte fragmentation.** Internal waste is `row_capacity - requested_rows`
   (`SourceExtentLease.internal_fragmentation_rows`). External waste is free
   pages not contiguous with each other; the per-physical-slot rule adds
   boundary waste even when the flat allocation has a contiguous run.
   `free_slot_count = min(logical_free, free_page_count)`
   (`dt_feature_p2p.py:341-353`) bounds publication count, not the existence of
   a requested contiguous extent; admission can pass and `publish()` still
   raise "all bounded feature source slots are leased" (`dt_feature_p2p.py:489-491`).
2. **Logical identity pressure.** Every live publication consumes one logical
   `source_slot_id` even when it is one page. Paged mode caps identities at
   `min(total_pages, 1024)` for the generation table
   (`dt_feature_p2p.py:263-271`), and a publication pins its identity until all
   transfers ack. Admission sees `free_slot_count`, not how many extents of the
   next wave's size exist.
3. **Slow-consumer pinning.** One `SourceExtentLease` is shared by every
   transfer in the publication. Same-host ring proof is per transfer
   (`target_feature_read_proof`, `dt_target.py:9363-9431`); remote retires on
   `SourceReadDone` or a candidate tombstone (`dt_target.py:7432`), but
   `read_done()` frees the page extent only after the last transfer. One slow
   Draft pins all rows and pages of the publication.
4. **Control-path delay.** Reserve-to-visible includes
   `reserve -> publish/publish_reserved -> producer event -> descriptor ->
   record visible`. Visible-to-proof includes Draft copy/READ, proof
   publication, and Target polling. Scheduler admission
   (`dt_target.py:4150-4181`, `:5271-5289`) and empty-prefill arbitration
   (`dt_target.py:4183`) can delay prefill or verify before any allocator call.

## 3. Measurements that decide the design

No proposal below should be implemented before these four measurements are
joined by publication (`slot_id`, `slot_generation`) and, where available, by
`draft_session_id`. The existing P0 joiner (`scripts/analyze_dt_target_p0.py`)
already correlates `feature_published`, `record_visible`, `descriptor_*`, and
`freeze_*`.

### M1. `effective_free <= 1` stall wall time

Primary signal: `target_feature_source_stall` start/end duration
(`dt_target.py:4150-4181`), emitted when
`free_slots <= deferred_feature_stages`. Corroborate with
`target_empty_prefill_arbitration` decisions `hard_capacity`/`one_slot_headroom`
and `target_ring_read_stall` (`dt_target.py:9383-9397`). Report wall time,
count, and the distribution of `free_slots - deferred_feature_stages`. The
stall event covers `effective_free <= 0`; measure the `<= 1` band by
timestamping the `should_defer_prefill` headroom branch
(`dt_target.py:5277-5284`). The event does **not** prove the allocator was the cause: it fires on the
scheduler gate, not on `reserve()` failure. Add a `source_reserve_failed`
observation carrying free pages, largest free extent, and requested pages;
without it, a high M1 cannot select any allocator design.

### M2. Reserve-to-read-proof lifetime split

For each publication/transfer record, add `reserve_ns` in
`_reserve_slot_locked()`/`TargetFeatureSourcePool.reserve()` (also when
`publish_reserved()` reuses a reservation) and `release_ns` when
`pool.read_done()` frees the slot. Reuse `descriptor_published_ns`
(`dt_feature_p2p.py:144-155`), `target_p0_hop.record_visible`, same-host
`target_feature_read_proof.observed_ns`, and remote
`target_source_done_batch_apply` or `_consume_remote_source_proof`
(`dt_target.py:7432`). Split lifetime into `reserve -> descriptor/record
visible` and `visible -> proof -> release`: the second half is consumer/pin
time, the first half is the Target control path.

### M3. Requested vs allocated rows and slowest-transfer pin tail

`SourceExtentLease` already has `requested_rows`, `row_capacity`, and
`internal_fragmentation_rows`. Descriptors carry only `nbytes`, so add an
experiment-only publication record with `requested_rows`, `allocated_rows =
page_count * page_rows`, `waste_rows`, free pages and largest free extent
before/after reservation, transfer count, and `max(transfer.nbytes)`.
Pin tail per publication is `release_ns - max(proof_ns)` and
`max(proof_ns) - min(proof_ns)`, recorded per owner (`draft_session_id`) too.
High waste with a small tail selects allocation policy; low waste with a large
tail selects lifetime/owner isolation.

### M4. Source-gate delayed prefill/verify counts and time

Count and time the scheduler paths that consult the source gate:
`scheduler.py:3282-3305` (`has_feature_source_capacity` false, plan idle),
`:3332-3338` and `:3438-3450` (`should_defer_prefill`, `one_slot_headroom`),
and `:3653-3658` (new waiting requests deferred). Join them with
`target_empty_prefill_arbitration`, `target_empty_prefill_ready_overlap`, and
`frozen_for_verify` counters `source_slots_free/total`,
`source_pages_free/total`, `source_largest_extent_pages`
(`dt_target.py:5213-5266`). Report delays separately for prefills and verify
waves; a gate delay with healthy free pages and largest extent points at
accounting or control, not allocator policy.

## 4. Alternatives

Only three alternatives are considered. Each states which M1-M4 selects or
rejects it, integration points, interface changes, same-host/cross-host
behavior, invariants, rollback, costs, and the smallest Target-only
experiment. Draft/Coordinator changes require owner approval.

### A. True fixed-page global pool with logical page lists and scatter/gather

**Design.** One global page pool over the flat allocation. A logical
publication owns a list of page IDs; each transfer owns one or more contiguous
runs within that list; page or owner refcounts decide when a page becomes free.
Descriptors carry a run list (or one descriptor per run). Same-host copies one
segment per run and coalesces adjacent physical runs. Cross-host requires one
READ per run, a gather into the Draft landing/destination, and a proof that
covers every run.

**Current integration points and required changes.**

- Target: replace `TargetSourcePageAllocator` with a global page map; change
  `_FeatureSlot` (`dt_feature_p2p.py:194-205`) to hold a page list;
  `FeatureSourceDescriptor` (`dt_feature_p2p.py:85-117`) must gain run
  information; `publication_offset()` (`dt_feature_p2p.py:326-338`) is no
  longer a single base for a publication.
- Same-host: `FeatureSourceDescriptor.source_ref_kwargs()` and
  `dt_protocol.SourceRef` must carry runs; `_accept_ring_task`
  (`dt_draft_engine.py:2640-2660`) and CUDA-IPC copy construction must accept
  multiple runs; `CudaIpcRuntime.copy_batch_async` already takes segments, but
  the descriptor cannot currently express them.
- Ring/native: v2's single `source_byte_offset`/`nbytes` record
  (`csrc/dt_ring_worker.cpp:464-516`) must become a v3 run list or multiple
  records per transfer; the worker must enqueue one READ/copy per run.
- Remote: `TargetRemoteRingWriter` (`dt_rdma.py:1672-1800`) and the remote
  worker's `serve_remote_record` need run lists; `DraftRdmaPuller` needs
  multi-run landing/gather.
- Coordinator: must relay and fence the new SourceRef version. This is a
  protocol change: **requires owner approval and Draft/Coordinator work.**

**Same-host vs cross-host.** Same-host is easier: multiple CUDA-IPC segments
can be enqueued on one stream, and `coalesce_page_runs` /
`build_copy_segments` already exist for the KV path and are a useful
implementation reference. Cross-host is harder: multiple WRs, ordering, and a
gather step; remote readiness remains per logical publication, and there is no
early READ proof on the remote ring (`dt_rdma.py:1683-1687`).

**Invariants and failure handling.** The run list must exactly cover every
published row once, with no hole or overlap. `source_slot_id`/generation must
still fence the READY identity; `source_byte_offset` is authoritative for
addresses; a stale generation must be rejected. A transfer is retired only
after every one of its runs is proven. If any CUDA copy or RDMA READ is
uncertain, quarantine every page in the publication, not only the failed run.
Cancellation waits for producer completion as today
(`dt_feature_p2p.py:811-918`). No re-publication after a ring kernel has
launched.

**Costs.** Memory is unchanged. CPU cost is O(runs) per allocation/release and
per copy plan. GPU cost is one copy enqueue/event per run instead of one per
transfer. RDMA cost is one WR per run instead of one per record, plus a gather
on the Draft. Coalescing adjacent runs is essential; without it, many 64-row
pages can multiply copy/WR count.

**Selection.** Select A only if M3 shows a material external-fragmentation or
run-eligibility problem (`largest_free_extent_pages` below the requested page
count, or reservations satisfiable only by multiple runs), M2's pin tail is not
dominant, and M1 correlates with reserve failures. M4 must tie delayed
prefill/verify to those failures. Reject A if the P0 high-water stays tiny, if
requested and allocated rows are already close, if M2 shows consumer/control
delay, or if M4 shows gate delay without reserve failures.

**Smallest Target-only experiment.** Full A is not a safe 300-800-line
Target-only experiment and is not proposed here. The smallest approximation is
a **global flat free-page list** that allows one contiguous run to cross old
physical-slot boundaries; it keeps one contiguous `SourceRef` per transfer and
therefore needs no Draft change. Ring v2 and the inline path already accept an
arbitrary contiguous range within the whole allocation
(`dt_ring_worker.cpp:468-484`, `dt_draft_engine.py:10072-10079`). Replace the
per-slot `_PageArena` boundary with a flat page map and a lease token that
replaces the arena-epoch check, keep one extent per lease, and add the M1-M3
counters. Expect roughly 300-450 production lines, default-off behind a new
Target env knob, with the existing allocator as the rollback path. If the
experiment shows transfers actually need multiple non-contiguous runs, stop:
there is no safe Target-only full-A experiment; request owner approval.

### B. Current-protocol-compatible size-class slab/TLSF allocator

**Design.** Keep one contiguous allocation, one logical READY identity per
publication, and one contiguous extent per lease. Replace the best-fit linear
page scan with size-class free lists (for example 1, 2, 4, 8 page slabs) or a
TLSF two-level bitmap over the existing per-slot arenas. Allocation rounds up
to a class; release coalesces and returns the run to its class. No descriptor,
ring, RDMA, Draft, or Coordinator change.

**Current integration points.** `TargetSourcePageAllocator.allocate/release`
(`dt_source_page_allocator.py:81-164`), its properties used by
`frozen_for_verify` (`dt_target.py:5213-5266`), and
`TargetFeatureSourcePool._reserve_slot_locked/_free_slot_locked`
(`dt_feature_p2p.py:367-432`). `SourceExtentLease`, `_FeatureSlot`,
`FeaturePublish`, and all wire types stay unchanged. `free_slot_count`
semantics should be documented and, for the experiment, supplemented by
`largest_free_extent_pages` so a future admission fix can use it.

**Same-host vs cross-host.** Identical. Nothing outside the allocator sees the
change: same bytes, same descriptors, same ring records, same RDMA ranges.

**Invariants and failure handling.** The physical-slot epoch and stale-release
checks stay exactly as today. Size-class rounding must not change the
all-or-nothing `active_transfers` rule; quarantine still retains the extent.
A class-rounding request larger than the slot is rejected as before. No CUDA
work is introduced, so rollback is simply the old allocator.

**Costs.** O(1) allocation/release and small per-class metadata; internal waste
up to class rounding; no GPU or RDMA cost. This is the lowest-risk policy
change.

**Selection.** Select B if M3 shows external fragmentation or reserve failures
with free pages and a small pin tail, while M1 correlates with reserve failures
rather than one slow owner. Internal waste helps only if the chosen classes
round below the current 64-row page. M4 must tie delayed prefill/verify to
extent-limited reservations. Reject B if the largest extent is never limiting,
the high-water is tiny, M2's pin tail dominates, or M4 delays occur without
reserve failures.

**Smallest experiment.** Allocator-only mode switch, default-off, roughly
250-400 production lines including statistics and startup validation. Run it
through the existing `scripts/dt_target_combo_service.sh` arm pattern with the
same Draft settings and `workload_fingerprint` check in
`scripts/summarize_dt_target_combo.py`. Add an arm differing from `early_page64`
only in allocator mode, with `DT_TARGET_EARLY_READY_RESULT` fixed.

### C. Per-Draft-owner pools/rings with page credits and independent reclamation

**Design.** Assign source capacity to an owner, where owner is the current
binding `(draft_instance_id, draft_session_id)` from `_draft_binding`
(`dt_target.py:771-772`). Full C gives each owner its own registered pool,
page arena, ready table, and ring, with a credit limit; only that owner's
transfers pin its pages. A mixed-owner verify wave is split into per-owner
publications so one slow owner cannot hold another owner's pages.

**Current integration points and required changes.**

- Target: `TargetFeatureSourcePool` is a singleton
  (`dt_target.py:892-915`), and `_feature_source_pool` is used directly by
  publish, reserve, poll, free, quarantine, and shutdown
  (`dt_target.py:1234-1455`, `:2269-2332`, `:3517-3594`, `:6425`, `:7698`).
  The per-owner policy would select a pool by `_draft_binding` at stage time.
  The `poll` packet currently broadcasts one `feature_source_free_slots`
  (`dt_target.py:3670-3673`); per-owner capacity needs an aggregate and an
  owner-aware admission rule while keeping TP lockstep.
- Same-host: per-session local ring writers already exist (`_ring_writers`,
  `_direct_ring_writer`, `dt_target.py:8970-9008`), and the Draft IPC mapping
  cache is keyed by `(peer, session, pool_generation, ipc_handle)`. Multiple
  pools are thus representable, but descriptor routing, owner credit
  accounting, and per-owner close/revocation are new Target work.
- Cross-host: `TargetRdmaEndpoint` exposes one registered pool and one ready
  table, and `DraftRdmaPuller` is "one Draft to one remote Target"
  (`dt_rdma.py:1121-1175`). Per-owner RDMA pools/rings require a multi-pool
  endpoint, ready-record routing, and Draft puller changes: **requires owner
  approval and Draft/Coordinator work.**
- Coordinator: per-owner publications still travel on existing task routing,
  but multi-pool source references and per-owner fencing need an agreed
  protocol. Label as future work.

**Same-host vs cross-host.** Same-host can realize most of the isolation with
one registered allocation carved into owner buckets; only the logical
publications and reclaim policy become owner-scoped. Cross-host must either
multiplex several pools behind one endpoint or run separate ready/pull
programs; neither is a local Target-only change.

**Invariants and failure handling.** Each owner's generation table and
read-proof fences must remain independent. The `_revoke_ring_owner` path
(`dt_target.py:9467`) already revokes one session's outstanding ring entries;
per-owner pools should reuse that notion, retiring only that owner's ranges and
quarantining only its pages. Removing an owner must not free another owner's
pages. TP lockstep still requires every rank to agree on the aggregate credits
and on which owner may publish; the source of truth remains the export rank's
pool state broadcast in `poll`.

**Costs.** CPU: O(owners) bookkeeping and one publication setup per owner in a
mixed wave. GPU: no extra bytes if owner row ranges are disjoint, but extra
copy enqueues/events and extra logical identities; duplicating a packed wave
per owner would add bytes and is not acceptable without measurement. RDMA:
more ring/ready state and potentially more QPs/WRs per owner. Memory is
unchanged if buckets are carved from the same allocation.

**Selection.** Select C if M2/M3 show the slowest-transfer pin tail is dominant,
requested rows are close to allocated, M1 stalls correlate with one
owner/session, page headroom is ample, and M4 shows that owner delaying the gate
while another owner has capacity. Reject C if M3 shows fragmentation or
largest-extent failures, if M4 delays are global with ample credit, or if
mixed-owner waves would require material byte duplication.

**Smallest Target-only experiment.** Keep the current single registered
allocation and wire format. Add per-owner page credits and split each wave's
publication into owner-scoped logical publications, allocating rows for each
owner's transfers from that owner's bucket. Every publication still uses one
contiguous extent and the existing per-transfer `SourceRef` format, so no Draft
change is needed; the Target consumes the existing per-transfer `read_done` and
ring proofs.
Default-off behind a Target env knob. Expect roughly 350-550 production lines.
The experiment must log per-owner stall time and pin tail and must prove that
TP ranks agree on the aggregate. Full per-owner pools, especially remote RDMA
pools, are not a 300-800-line Target-only experiment; they require owner
approval.

## 5. Correctness and rollback gates

Apply to any chosen experiment:

- **Generation and identity.** Never reuse a logical slot or page extent until
  the exact `(slot_id, slot_generation)` has a matching read proof or an
  authenticated retirement proof. A stale slot id, generation, session, or
  pool generation must raise, not free.
- **Read-proof-before-reuse.** Same-host ring uses the INPUT_READY proof
  (`target_feature_read_proof`); non-ring uses `SourceReadDone`; remote ring
  uses `SourceReadDone` or the authenticated candidate tombstone. A cancellation
  is not a read proof and must wait for the producer event before reuse
  (`dt_feature_p2p.py:811-918`).
- **Ownership and exact tokens.** Keep `_source_slot_by_transfer`,
  `_candidate_arena`, request/KV ownership, and the descriptor
  `nbytes`/`row_count`/token mapping unchanged. Any experiment that changes
  rows copied or token coverage is rejected.
- **TP lockstep.** The export rank remains the source of allocator state; every
  rank consumes the same broadcast free/credit packet from `poll`
  (`dt_target.py:3655-3700`). No rank-local allocator decision may diverge.
- **Quarantine and close.** An uncertain CUDA copy/READ quarantines the whole
  affected lease; `close()` must still count active leases and reservations
  (`dt_feature_p2p.py:945-971`). A new allocator must not make shutdown look
  drained while pages are still referenced.
- **Default-off and rollback.** The current allocator and page64 path remain
  compiled and selectable. A startup validation must prove the experiment is
  inert when the knob is off, and the first failing invariant must fall back to
  the current allocator or quarantine, never to a partial free. Any experiment
  that requires a Draft or Coordinator change is out of scope for this task.
- **Stop conditions for the A/B services.** Stop an arm on any stale-proof
  acceptance, any free-before-proof observation, any TP free-slot divergence,
  any change in accepted tokens, or any quarantine count increase not explained
  by an injected failure.

## 6. Decision table

The table is intentionally outcome-first: build one thing only after its row is
selected by measurements. Thresholds are pre-registered (for example, a stall
is material if it exceeds 1% of run wall time; waste is material if
`waste_rows / allocated_rows >= 25%`; a pin tail is material if it is at least
50% of the lease lifetime; high-water is meaningful only if it approaches the
page count). Adjust the numeric thresholds before the run, not after seeing it.

| Measured outcome | One implementation to build |
|---|---|
| M1 stall wall is near zero, M3 high-water is a tiny fraction of pages, and no reserve-failure observation exists | None. Keep page64; fix no allocator. Publish the measurement and close the allocator hypothesis. |
| M1/M4 stalls occur with ample free pages and a sufficient largest extent, while M2 reserve-to-publish is small and M3 pin tail is small | None of A/B/C. Keep page64 and document the gate-accounting discrepancy for a separate issue; do not build an allocator here. |
| M1 correlates with `source_reserve_failed`; M3 shows `largest_free_extent_pages` below the request or material external waste; M3 pin tail is small | Build **B**: per-slot size-class slab/TLSF, contiguous extents, no Draft change. |
| B is already in place or is rejected by internal-waste data, but M3 still shows external waste or requests that would fit only across old physical-slot boundaries; M2 pin tail is small | Build the **A Target-only approximation**: global flat free-page list with one contiguous run per lease, no scatter, no Draft change. |
| M2/M3 show the slowest-transfer pin tail dominates, requested and allocated rows are close, free pages are ample, and M1 stalls correlate with one `draft_session_id` | Build **C Target-only approximation**: owner-scoped publications and per-owner page credits inside the current single allocation. |
| M2 shows reserve-to-descriptor or descriptor-to-proof control delay dominates with healthy allocator values, or M1 stalls occur when the gate is low but pages and extents are healthy | None of A/B/C. Keep the allocator and document the control-path finding; do not build an allocator here. |
| Full A is required because M3 proves real multi-run transfers, or full C is required because remote per-owner pools are mandatory | Stop. These require owner approval plus Draft/Coordinator protocol work and are outside this task; do not implement a Target-only substitute that silently changes the wire. |
| No row is selected by the measurements | No implementation. Page64 is not the bottleneck unless the measurements say it is. |
