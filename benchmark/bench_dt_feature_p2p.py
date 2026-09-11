"""Two-process H800 gate for the DT DFlash feature transport.

This benchmark intentionally exercises ``TargetFeatureSourcePool`` and
``DraftFeaturePullTransport`` rather than the Draft-KV migration transport.  A
small queue-based control plane carries descriptors and read-done metadata;
the hidden-state payload moves directly between CUDA allocations through the
production feature transport.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import socket
import sys
import threading
import time
import traceback
import uuid
from array import array
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT / "sglang" / "python", _ROOT / "dt-runtime" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402
from dt_protocol import (  # noqa: E402
    CancelTransfers,
    DraftKvBuffer,
    DraftKvMigrationExport,
    DraftKvMigrationPrepare,
    DraftKvPool,
    PROTOCOL_VERSION,
    SourceReadDone,
)
from sglang.srt.speculative.dt_feature_p2p import (  # noqa: E402
    DraftFeaturePullTransport,
    FeatureTransferSlice,
    FeatureTransportError,
    TargetFeatureSourcePool,
)


def _now_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


class _RecordingOutbox:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str, object]] = []

    def add(self, operation, message_id, message) -> None:
        self.entries.append((str(operation), message_id, message))


def _target_shell(pool: TargetFeatureSourcePool, config: dict[str, Any]):
    from sglang.srt.speculative.dt_target import DTTargetController

    target = DTTargetController.__new__(DTTargetController)
    target.instance_id = config["target_instance_id"]
    target.is_dflash = True
    target._feature_source_pool = pool
    target._source_slot_by_transfer = {}
    target._source_deadline_by_transfer = {}
    target._source_done_applied_ids = set()
    target._cancelled_transfers = set()
    target._client = SimpleNamespace(
        coordinator_session_id=config["coordinator_session_id"]
    )
    target._delivery_outbox = _RecordingOutbox()
    target._flush_reliable_outbox = lambda: None
    target._profile_emit = lambda *_args, **_kwargs: None
    return target


def _source_worker(
    config: dict[str, Any],
    source_session_id: str,
    generation_index: int,
    command_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    try:
        from sglang.srt.speculative.dt_target import DTTargetController

        torch.cuda.set_device(config["source_gpu"])
        device = torch.device("cuda", config["source_gpu"])
        pool = TargetFeatureSourcePool(
            target_instance_id=config["target_instance_id"],
            source_session_id=source_session_id,
            device=device,
            slot_count=1,
            max_tokens=config["rows"],
            feature_width=config["feature_width"],
        )
        target = _target_shell(pool, config)
        properties = torch.cuda.get_device_properties(device)
        result_queue.put(
            {
                "kind": "source_ready",
                "pid": os.getpid(),
                "started_ns": _now_ns(),
                "generation_index": generation_index,
                "source_session_id": pool.source_session_id,
                "pool_generation": pool.pool_generation,
                "transport_session": pool.transport_session,
                "device_uuid": pool.device_uuid,
                "device_name": properties.name,
                "device_total_memory": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
                "cuda_runtime_version": pool._runtime.runtime_version,
            }
        )
        while True:
            command = command_queue.get()
            action = command["action"]
            if action == "publish":
                transfer_id = command["transfer_id"]
                value = int(command["value"])
                rows = int(command.get("rows", config["rows"]))
                hidden = torch.full(
                    (rows, config["feature_width"]),
                    value,
                    dtype=torch.bfloat16,
                    device=device,
                )
                published = pool.publish(
                    hidden_states=hidden,
                    feature_schema_id=config["feature_schema_id"],
                    transfers=(
                        FeatureTransferSlice(
                            transfer_id=transfer_id,
                            row_offset=0,
                            row_count=rows,
                        ),
                    ),
                )
                descriptor = published.descriptors[0]
                target._source_slot_by_transfer[transfer_id] = (
                    descriptor.source_slot_id,
                    descriptor.slot_generation,
                )
                target._source_deadline_by_transfer[transfer_id] = _now_ns() + int(
                    config["timeout_s"] * 1_000_000_000
                )
                result_queue.put(
                    {
                        "kind": "published",
                        "descriptor": descriptor,
                        "value": value,
                        "published_ns": published.descriptor_published_ns,
                        "free_slots": pool.free_slot_count,
                        "quarantined_slots": pool.quarantined_slot_count,
                    }
                )
                continue
            if action == "probe_publish":
                blocked = False
                error = ""
                probe_transfer = command["transfer_id"]
                try:
                    hidden = torch.full(
                        (config["rows"], config["feature_width"]),
                        int(command["value"]),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    probe = pool.publish(
                        hidden_states=hidden,
                        feature_schema_id=config["feature_schema_id"],
                        transfers=(
                            FeatureTransferSlice(
                                transfer_id=probe_transfer,
                                row_offset=0,
                                row_count=config["rows"],
                            ),
                        ),
                    )
                except FeatureTransportError as exc:
                    blocked = True
                    error = str(exc)
                else:
                    descriptor = probe.descriptors[0]
                    pool.cancel(
                        transfer_id=descriptor.transfer_id,
                        source_slot_id=descriptor.source_slot_id,
                        slot_generation=descriptor.slot_generation,
                        source_session_id=descriptor.source_session_id,
                        pool_generation=descriptor.pool_generation,
                    )
                result_queue.put(
                    {
                        "kind": "probe_publish_result",
                        "blocked": blocked,
                        "error": error,
                        "observed_ns": _now_ns(),
                        "free_slots": pool.free_slot_count,
                        "quarantined_slots": pool.quarantined_slot_count,
                    }
                )
                continue
            if action == "expire":
                transfer_id = command["transfer_id"]
                target._source_deadline_by_transfer[transfer_id] = _now_ns() - 1
                DTTargetController._expire_source_leases(target, now_ns=_now_ns())
                result_queue.put(
                    {
                        "kind": "expired",
                        "transfer_id": transfer_id,
                        "observed_ns": _now_ns(),
                        "free_slots": pool.free_slot_count,
                        "quarantined_slots": pool.quarantined_slot_count,
                    }
                )
                continue
            if action == "read_done":
                descriptor = command["descriptor"]
                if command.get("message") is not None:
                    message = SourceReadDone(**command["message"])
                else:
                    message = SourceReadDone(
                        protocol_version=PROTOCOL_VERSION,
                        transfer_id=command.get("transfer_id", descriptor.transfer_id),
                        target_instance_id=config["target_instance_id"],
                        draft_instance_id=config["draft_instance_id"],
                        source_slot_id=command.get(
                            "source_slot_id", descriptor.source_slot_id
                        ),
                        slot_generation=command.get(
                            "slot_generation", descriptor.slot_generation
                        ),
                        completed_ns=_now_ns(),
                        instance_session_id=config["draft_session_id"],
                        coordinator_session_id=config["coordinator_session_id"],
                        source_session_id=command.get(
                            "source_session_id", descriptor.source_session_id
                        ),
                        pool_generation=command.get(
                            "pool_generation", descriptor.pool_generation
                        ),
                    )
                error = ""
                try:
                    DTTargetController._consume_source_read_done(target, message)
                except Exception as exc:  # noqa: BLE001 - evidence captures fencing
                    error = f"{type(exc).__name__}: {exc}"
                result_queue.put(
                    {
                        "kind": "read_done_result",
                        "transfer_id": message.transfer_id,
                        "slot_generation": message.slot_generation,
                        "error": error,
                        "applied": not error
                        and message.transfer_id not in target._source_slot_by_transfer,
                        "ack_outbox_entries": len(target._delivery_outbox.entries),
                        "observed_ns": _now_ns(),
                        "free_slots": pool.free_slot_count,
                        "quarantined_slots": pool.quarantined_slot_count,
                    }
                )
                continue
            if action == "status":
                result_queue.put(
                    {
                        "kind": "source_status",
                        "free_slots": pool.free_slot_count,
                        "quarantined_slots": pool.quarantined_slot_count,
                        "active_transfers": sorted(target._source_slot_by_transfer),
                        "observed_ns": _now_ns(),
                    }
                )
                continue
            if action == "stop":
                pool.close()
                result_queue.put(
                    {
                        "kind": "source_stopped",
                        "stopped_ns": _now_ns(),
                        "free_slots": pool.free_slot_count,
                        "quarantined_slots": pool.quarantined_slot_count,
                    }
                )
                return
            raise ValueError(f"unknown source command {action!r}")
    except Exception as exc:  # noqa: BLE001 - report child-process failure
        result_queue.put(
            {
                "kind": "error",
                "where": "source",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


class _InjectedFeatureRuntime:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._lock = threading.Lock()
        self._mode = "normal"
        self._delay_cycles = 0
        self._enqueued = threading.Event()
        self._enqueued_ns = 0

    def configure(self, *, mode: str, delay_cycles: int) -> None:
        with self._lock:
            self._mode = mode
            self._delay_cycles = delay_cycles
            self._enqueued.clear()
            self._enqueued_ns = 0

    def copy_batch_async(self, segments, stream) -> None:
        with self._lock:
            mode = self._mode
            delay_cycles = self._delay_cycles
        if delay_cycles:
            torch.cuda._sleep(delay_cycles)
        self._runtime.copy_batch_async(segments, stream)
        with self._lock:
            self._enqueued_ns = _now_ns()
            self._enqueued.set()
        if mode == "fail_after_enqueue":
            raise RuntimeError("injected post-enqueue feature-copy failure")

    def enqueue_state(self) -> tuple[bool, int]:
        with self._lock:
            return self._enqueued.is_set(), self._enqueued_ns


def _dflash_launch_liveness_gate(device: torch.device, delay_cycles: int) -> dict:
    from sglang.srt.speculative.dt_draft_engine import (
        FlashMTPDraftEngine,
        PendingDraftTask,
    )

    rows = 4
    feature_width = 16
    block_size = 8
    committed_len = 64
    engine = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
    engine.instance_id = "launch-liveness-draft"
    engine.device = device
    engine.profile = SimpleNamespace(block_size=block_size)
    engine._pending_dflash_completions = deque()
    engine._feature_staging_buffers = [
        torch.zeros((rows, feature_width), dtype=torch.bfloat16, device=device)
    ]
    state = SimpleNamespace(
        req=SimpleNamespace(req_pool_idx=7), valid_committed_len=committed_len
    )
    cache_loc = torch.arange(rows, dtype=torch.int64, device=device)
    engine._prepare_dflash_task = lambda _task: (state, cache_loc)
    captured: dict[str, Any] = {}

    class _DelayedMaterializer:
        @staticmethod
        def append(*, target_hidden, cache_loc, positions) -> None:
            captured["target_shape"] = tuple(target_hidden.shape)
            captured["cache_shape"] = tuple(cache_loc.shape)
            captured["positions_shape"] = tuple(positions.shape)
            torch.cuda._sleep(delay_cycles)

    candidates = torch.empty((1, block_size), dtype=torch.int32, device=device)

    def _proposal(
        _self,
        *,
        states,
        req_pool_indices,
        prefix_lens,
        seq_lens_cpu,
        bonus_tokens,
    ):
        captured["state_count"] = len(states)
        captured["req_pool_indices_device"] = req_pool_indices.device.type
        captured["prefix_lens_device"] = prefix_lens.device.type
        captured["seq_lens_cpu_device"] = seq_lens_cpu.device.type
        captured["seq_lens_cpu"] = seq_lens_cpu.tolist()
        captured["bonus_tokens_device"] = bonus_tokens.device.type
        return candidates

    engine._dflash_materializer = _DelayedMaterializer()
    engine._run_dflash_proposal = MethodType(_proposal, engine)
    task = SimpleNamespace(
        transfer_id="launch-liveness",
        request_id="launch-liveness",
        draft_kv_handle="launch-liveness",
        feature_token_offset=committed_len - rows,
        feature_token_count=rows,
        committed_kv_len=committed_len,
        round_id=1,
        deadline_ns=_now_ns() + 60_000_000_000,
        requires_proposal=True,
        next_token_id=42,
    )
    launched_ns = _now_ns()
    FlashMTPDraftEngine._launch_dflash_materialization(
        engine,
        pending=[PendingDraftTask(task=task, pivot_bytes=None)],
        staging_slot=0,
        row_ranges=((0, rows),),
    )
    launch_return_ns = _now_ns()
    completion = engine._pending_dflash_completions[0]
    event_pending_at_return = not completion.completion_event.query()
    poll_durations_ns = []
    scheduler_ticks = 0
    control_ticks = 0
    poll_deadline = time.monotonic() + 0.02
    while time.monotonic() < poll_deadline:
        if completion.completion_event.query():
            break
        poll_started_ns = _now_ns()
        FlashMTPDraftEngine._poll_dflash_completions(engine)
        poll_durations_ns.append(_now_ns() - poll_started_ns)
        if len(engine._pending_dflash_completions) != 1:
            raise RuntimeError(
                "pending DFlash completion was consumed before its event"
            )
        scheduler_ticks += 1
        control_ticks += 1
        time.sleep(0.0001)
    completion.completion_event.synchronize()
    completed_ns = _now_ns()
    materialization_gpu_ms = completion.materialize_start_event.elapsed_time(
        completion.materialize_done_event
    )
    engine._pending_dflash_completions.clear()
    launch_elapsed_ms = (launch_return_ns - launched_ns) / 1_000_000
    gate_passed = bool(
        event_pending_at_return
        and scheduler_ticks > 0
        and control_ticks > 0
        and captured.get("seq_lens_cpu_device") == "cpu"
        and captured.get("seq_lens_cpu") == [committed_len + block_size]
        and captured.get("req_pool_indices_device") == "cuda"
        and captured.get("prefix_lens_device") == "cuda"
        and captured.get("bonus_tokens_device") == "cuda"
        and launch_elapsed_ms < materialization_gpu_ms
    )
    return {
        "launch_submitted_ns": launched_ns,
        "launch_return_ns": launch_return_ns,
        "gpu_completed_ns": completed_ns,
        "launch_elapsed_ms": launch_elapsed_ms,
        "materialization_gpu_ms": materialization_gpu_ms,
        "event_pending_at_launch_return": event_pending_at_return,
        "scheduler_ticks_while_gpu_pending": scheduler_ticks,
        "unrelated_control_ticks_while_gpu_pending": control_ticks,
        "max_completion_poll_ns": max(poll_durations_ns, default=0),
        "metadata": captured,
        "gate_passed": gate_passed,
    }


def _dflash_two_batch_liveness_gate(device: torch.device, delay_cycles: int) -> dict:
    """Keep a second ready batch staged while the first GPU event is pending."""

    from sglang.srt.speculative.dt_draft_engine import (
        FlashMTPDraftEngine,
        PendingDraftTask,
        PendingFeatureBatch,
    )

    rows = 4
    feature_width = 16
    block_size = 8
    committed_len = 64
    engine = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
    engine.instance_id = "two-batch-draft"
    engine.device = device
    engine.profile = SimpleNamespace(block_size=block_size)
    engine._pending_dflash_completions = deque()
    engine._feature_staging_buffers = [
        torch.zeros((rows, feature_width), dtype=torch.bfloat16, device=device)
        for _ in range(2)
    ]
    states = {
        f"two-batch-{index}": SimpleNamespace(
            req=SimpleNamespace(req_pool_idx=7 + index),
            valid_committed_len=committed_len,
        )
        for index in range(2)
    }
    cache_locs = {
        transfer_id: torch.arange(rows, dtype=torch.int64, device=device)
        for transfer_id in states
    }
    engine._prepare_dflash_task = lambda task: (
        states[task.transfer_id],
        cache_locs[task.transfer_id],
    )
    captured: dict[str, Any] = {"materializer_calls": 0, "metadata": []}

    class _FirstBatchDelayedMaterializer:
        @staticmethod
        def append(*, target_hidden, cache_loc, positions) -> None:
            captured["materializer_calls"] += 1
            captured["last_shapes"] = {
                "target": tuple(target_hidden.shape),
                "cache": tuple(cache_loc.shape),
                "positions": tuple(positions.shape),
            }
            if captured["materializer_calls"] == 1:
                torch.cuda._sleep(delay_cycles)

    candidates = torch.empty((1, block_size), dtype=torch.int32, device=device)

    def _proposal(
        _self,
        *,
        states,
        req_pool_indices,
        prefix_lens,
        seq_lens_cpu,
        bonus_tokens,
    ):
        captured["metadata"].append(
            {
                "state_count": len(states),
                "req_pool_indices_device": req_pool_indices.device.type,
                "prefix_lens_device": prefix_lens.device.type,
                "seq_lens_cpu_device": seq_lens_cpu.device.type,
                "seq_lens_cpu": seq_lens_cpu.tolist(),
                "bonus_tokens_device": bonus_tokens.device.type,
            }
        )
        return candidates

    now_ns = _now_ns()
    tasks = [
        SimpleNamespace(
            transfer_id=f"two-batch-{index}",
            request_id=f"two-batch-request-{index}",
            draft_kv_handle=f"two-batch-handle-{index}",
            source_read_migration_id="",
            feature_token_offset=committed_len - rows,
            feature_token_count=rows,
            committed_kv_len=committed_len,
            round_id=index,
            deadline_ns=now_ns + 60_000_000_000,
            requires_proposal=True,
            next_token_id=42 + index,
        )
        for index in range(2)
    ]

    def _ready_future() -> Future:
        future = Future()
        future.set_result(
            SimpleNamespace(
                submitted_ns=now_ns,
                gpu_completed_ns=now_ns,
                mapping_ns=0,
                driver_submit_ns=0,
                gpu_copy_ns=0,
                feature_bytes=rows * feature_width * 2,
                cold_mapping_count=0,
            )
        )
        return future

    engine._pending_feature_batches = deque(
        PendingFeatureBatch(
            pending=[PendingDraftTask(task=task, pivot_bytes=None)],
            staging_slot=index,
            row_ranges=((0, rows),),
            futures=(_ready_future(),),
            submitted_ns=now_ns,
        )
        for index, task in enumerate(tasks)
    )
    engine._dflash_materializer = _FirstBatchDelayedMaterializer()
    engine._run_dflash_proposal = MethodType(_proposal, engine)
    engine.batch_source_read_done = False
    engine._free_feature_staging_slots = deque()
    engine._cancelled_ids = set()
    engine._pending = deque()
    engine._published_by_batch = {}
    engine._publish_source_done = lambda _task: None
    engine._release_persistent_request = lambda _handle: None
    engine._publish_failures = lambda _failures: None
    engine._publish_successes = lambda _tasks, _candidate_bytes: None
    engine._publish_materialization_successes = lambda _tasks: None
    engine._publish_capacity = lambda: None
    engine._resume_feature_finalizations = lambda: None
    engine._flush_source_done = lambda: None
    engine._prepared_migrations = {}
    engine._migration_quiescence_probes = {}
    engine._migration_quiescence_retry_ns = {}
    engine._quarantined_migration_ids = set()
    engine._p2p_transport = None
    engine._deferred_migration_gpu_controls = {}
    engine._drain_deferred_persistent_release = lambda _handle: False

    class _ControlClient:
        def __init__(self) -> None:
            self.messages = deque()
            self.instance_session_id = "two-batch-draft-session"
            self.coordinator_session_id = "two-batch-coordinator-session"

        def recv_nowait(self):
            messages = tuple(self.messages)
            self.messages.clear()
            return messages

    engine._client = _ControlClient()

    first_poll_started_ns = _now_ns()
    FlashMTPDraftEngine._poll_feature_batches(engine)
    first_poll_returned_ns = _now_ns()
    first_completion = engine._pending_dflash_completions[0]
    first_event_pending = not first_completion.completion_event.query()
    second_ready_while_first_pending = bool(
        len(engine._pending_feature_batches) == 1
        and engine._pending_feature_batches[0].futures[0].done()
        and captured["materializer_calls"] == 1
    )

    guard_started_ns = _now_ns()
    guard_error = ""
    try:
        FlashMTPDraftEngine._launch_dflash_materialization(
            engine,
            pending=[PendingDraftTask(task=tasks[1], pivot_bytes=None)],
            staging_slot=1,
            row_ranges=((0, rows),),
        )
    except RuntimeError as exc:
        guard_error = str(exc)
    guard_returned_ns = _now_ns()

    engine._client.messages.append(
        CancelTransfers(
            protocol_version=PROTOCOL_VERSION,
            target_instance_id="unrelated-target",
            transfer_ids=("unrelated-control",),
            reason="two_batch_liveness_gate",
        )
    )
    scheduler_ticks = 0
    control_poll_durations_ns = []
    feature_poll_durations_ns = []
    completion_poll_durations_ns = []
    second_remained_staged = True
    deadline = time.monotonic() + 10
    while not first_completion.completion_event.query():
        if time.monotonic() >= deadline:
            raise TimeoutError("two-batch DFlash liveness gate timed out")
        poll_started_ns = _now_ns()
        FlashMTPDraftEngine._poll_messages(engine)
        control_poll_durations_ns.append(_now_ns() - poll_started_ns)
        poll_started_ns = _now_ns()
        FlashMTPDraftEngine._poll_dflash_completions(engine)
        completion_poll_durations_ns.append(_now_ns() - poll_started_ns)
        poll_started_ns = _now_ns()
        FlashMTPDraftEngine._poll_feature_batches(engine)
        feature_poll_durations_ns.append(_now_ns() - poll_started_ns)
        second_remained_staged = second_remained_staged and bool(
            len(engine._pending_feature_batches) == 1
            and captured["materializer_calls"] == 1
        )
        scheduler_ticks += 1
        time.sleep(0.0001)

    first_event_ready_ns = _now_ns()
    FlashMTPDraftEngine._poll_dflash_completions(engine)
    if engine._pending_dflash_completions:
        raise RuntimeError("first DFlash completion did not drain after readiness")
    second_poll_started_ns = _now_ns()
    FlashMTPDraftEngine._poll_feature_batches(engine)
    second_poll_returned_ns = _now_ns()
    if len(engine._pending_dflash_completions) != 1:
        raise RuntimeError("second ready feature batch was not submitted")
    second_completion = engine._pending_dflash_completions[0]
    second_completion.completion_event.synchronize()
    FlashMTPDraftEngine._poll_dflash_completions(engine)

    first_poll_ms = (first_poll_returned_ns - first_poll_started_ns) / 1_000_000
    second_poll_ms = (second_poll_returned_ns - second_poll_started_ns) / 1_000_000
    first_gpu_ms = first_completion.materialize_start_event.elapsed_time(
        first_completion.materialize_done_event
    )
    gate_passed = bool(
        first_event_pending
        and second_ready_while_first_pending
        and second_remained_staged
        and scheduler_ticks > 0
        and "unrelated-control" in engine._cancelled_ids
        and "must wait for the prior completion event" in guard_error
        and (guard_returned_ns - guard_started_ns) < 10_000_000
        and max(feature_poll_durations_ns, default=0) < 10_000_000
        and max(control_poll_durations_ns, default=0) < 10_000_000
        and captured["materializer_calls"] == 2
        and len(captured["metadata"]) == 2
        and all(
            item["seq_lens_cpu"] == [committed_len + block_size]
            for item in captured["metadata"]
        )
        and second_poll_started_ns >= first_event_ready_ns
        and second_poll_ms < first_gpu_ms / 4
        and not engine._pending_feature_batches
        and not engine._pending_dflash_completions
    )
    return {
        "first_poll_ms": first_poll_ms,
        "first_gpu_ms": first_gpu_ms,
        "first_event_pending_at_poll_return": first_event_pending,
        "second_ready_while_first_pending": second_ready_while_first_pending,
        "second_remained_staged": second_remained_staged,
        "second_poll_ms": second_poll_ms,
        "second_submitted_after_first_ready": (
            second_poll_started_ns >= first_event_ready_ns
        ),
        "scheduler_ticks_while_first_pending": scheduler_ticks,
        "control_message_processed_while_first_pending": (
            "unrelated-control" in engine._cancelled_ids
        ),
        "max_control_poll_ns": max(control_poll_durations_ns, default=0),
        "max_completion_poll_ns": max(completion_poll_durations_ns, default=0),
        "max_feature_poll_ns": max(feature_poll_durations_ns, default=0),
        "direct_launch_guard_ms": (guard_returned_ns - guard_started_ns) / 1_000_000,
        "direct_launch_guard_error": guard_error,
        "materializer_calls": captured["materializer_calls"],
        "metadata": captured["metadata"],
        "gate_passed": gate_passed,
    }


def _dflash_migration_control_liveness_gate(
    device: torch.device, delay_cycles: int
) -> dict:
    """Defer real prepare/export controls behind unrelated DFlash GPU work."""

    from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.dt_draft_engine import (
        FlashMTPDraftEngine,
        PendingDraftTask,
        PersistentDraftRequest,
    )

    block_size = 8
    committed_len = 64
    feature_rows = 4
    feature_width = 16

    class _MigrationControlClient:
        def __init__(self, instance_id: str) -> None:
            self.messages = deque()
            self.instance_session_id = f"{instance_id}-session"
            self.coordinator_session_id = "migration-control-coordinator-session"

        def recv_nowait(self):
            messages = tuple(self.messages)
            self.messages.clear()
            return messages

    class _ProducerTransport:
        def __init__(self, instance_id: str) -> None:
            self.pool_generation = f"{instance_id}-pool-generation"

        @staticmethod
        def producer_event():
            event = torch.cuda.Event(interprocess=True)
            event.record(torch.cuda.current_stream(device))
            return event, event.ipc_handle()

    class _DelayedMaterializer:
        @staticmethod
        def append(*, target_hidden, cache_loc, positions) -> None:
            del target_hidden, cache_loc, positions
            torch.cuda._sleep(delay_cycles)

    def _build_engine(instance_id: str):
        engine = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
        engine.instance_id = instance_id
        engine.device = device
        engine.profile = SimpleNamespace(
            profile_id="dflash-migration-liveness",
            draft_k=block_size,
            block_size=block_size,
        )
        engine._is_dflash = True
        engine.migration_backend = "cuda-ipc"
        engine._client = _MigrationControlClient(instance_id)
        engine._p2p_transport = _ProducerTransport(instance_id)
        kv_cache = SimpleNamespace(
            use_hnd=False,
            is_quantized_kv_cache=False,
        )
        engine._draft_model_runner = SimpleNamespace(
            server_args=SimpleNamespace(page_size=1),
            req_to_token_pool=ReqToTokenPool(8, 256, device, False),
            token_to_kv_pool_allocator=TokenToKVPoolAllocator(
                1024,
                torch.bfloat16,
                device,
                kv_cache,
                False,
            ),
        )
        engine._kv_pool = DraftKvPool(
            transport="cuda-ipc",
            instance_session_id=engine._client.instance_session_id,
            pool_generation=engine._p2p_transport.pool_generation,
            transport_session=f"{instance_id}-transport-session",
            profile_id=engine.profile.profile_id,
            layout="NHD",
            page_size=1,
            device_id=int(device.index),
            device_uuid=str(torch.cuda.get_device_properties(device).uuid),
            buffers=(
                DraftKvBuffer(
                    allocation_index=0,
                    allocation_offset=0,
                    visible_length=1024 * 16,
                    item_length=16,
                ),
            ),
        )
        engine._delivery_outbox = SimpleNamespace(
            get=lambda *_args: None,
            retry_now=lambda *_args: None,
        )
        engine._flush_source_done = lambda: None
        engine._flush_reliable_outbox = lambda: None
        engine._drop_cancelled_pending = lambda: None
        engine._publish_capacity = lambda: None
        engine._publish_successes = lambda _tasks, _candidate_bytes: None
        engine._publish_materialization_successes = lambda _tasks: None
        engine._publish_failures = lambda _failures: None
        engine._publish_source_done = lambda _task: None
        engine._drain_deferred_persistent_release = lambda _handle: False
        engine._pending = deque()
        engine._pending_ids = set()
        engine._received_ns_by_transfer = {}
        engine._cancelled_ids = set()
        engine._published_by_batch = {}
        engine._persistent_requests = {}
        engine._prepared_migrations = {}
        engine._deferred_migration_gpu_controls = {}
        engine._migration_roles = {}
        engine._migration_decisions = {}
        engine._migration_release_requests = {}
        engine._migration_cleanup_ids = {}
        engine._migration_source_events = {}
        engine._migration_source_handles = {}
        engine._migration_quiescence_probes = {}
        engine._migration_quiescence_retry_ns = {}
        engine._quarantined_migration_ids = set()
        engine._pending_feature_batches = deque()
        engine._pending_dflash_completions = deque()
        engine._free_feature_staging_slots = deque()
        engine._feature_staging_buffers = [
            torch.zeros(
                (feature_rows, feature_width),
                dtype=torch.bfloat16,
                device=device,
            )
        ]
        engine.batch_source_read_done = False
        statuses = []
        engine._publish_migration_status = statuses.append
        return engine, statuses

    def _launch_unrelated_dflash(engine, label: str):
        state = SimpleNamespace(
            req=SimpleNamespace(req_pool_idx=7),
            valid_committed_len=committed_len,
        )
        cache_loc = torch.arange(feature_rows, dtype=torch.int64, device=device)
        engine._prepare_dflash_task = lambda _task: (state, cache_loc)
        engine._dflash_materializer = _DelayedMaterializer()
        candidates = torch.empty((1, block_size), dtype=torch.int32, device=device)

        def _proposal(
            _self,
            *,
            states,
            req_pool_indices,
            prefix_lens,
            seq_lens_cpu,
            bonus_tokens,
        ):
            del states, req_pool_indices, prefix_lens, seq_lens_cpu, bonus_tokens
            return candidates

        engine._run_dflash_proposal = MethodType(_proposal, engine)
        task = SimpleNamespace(
            transfer_id=f"unrelated-{label}",
            request_id=f"unrelated-request-{label}",
            draft_kv_handle=f"unrelated-handle-{label}",
            source_read_migration_id="",
            feature_token_offset=committed_len - feature_rows,
            feature_token_count=feature_rows,
            committed_kv_len=committed_len,
            round_id=1,
            deadline_ns=_now_ns() + 60_000_000_000,
            requires_proposal=True,
            next_token_id=42,
        )
        FlashMTPDraftEngine._launch_dflash_materialization(
            engine,
            pending=[PendingDraftTask(task=task, pivot_bytes=None)],
            staging_slot=0,
            row_ranges=((0, feature_rows),),
        )
        return engine._pending_dflash_completions[0]

    def _poll_control_while_pending(engine, completion, message, cancel_id: str):
        engine._client.messages.extend(
            (
                message,
                CancelTransfers(
                    protocol_version=PROTOCOL_VERSION,
                    target_instance_id="unrelated-target",
                    transfer_ids=(cancel_id,),
                    reason="migration_control_liveness_gate",
                ),
            )
        )
        first_started_ns = _now_ns()
        FlashMTPDraftEngine._poll_messages(engine)
        first_elapsed_ns = _now_ns() - first_started_ns
        pending_after_first_poll = not completion.completion_event.query()
        deferred_after_first_poll = len(engine._deferred_migration_gpu_controls)
        status_count_while_pending = 0
        control_poll_ns = [first_elapsed_ns]
        scheduler_ticks = 1
        deadline = time.monotonic() + 10
        while not completion.completion_event.query():
            if time.monotonic() >= deadline:
                raise TimeoutError("migration control liveness gate timed out")
            started_ns = _now_ns()
            FlashMTPDraftEngine._poll_messages(engine)
            control_poll_ns.append(_now_ns() - started_ns)
            status_count_while_pending = max(
                status_count_while_pending,
                int(not engine._deferred_migration_gpu_controls),
            )
            scheduler_ticks += 1
            time.sleep(0.0001)
        ready_ns = _now_ns()
        FlashMTPDraftEngine._poll_dflash_completions(engine)
        if engine._pending_dflash_completions:
            raise RuntimeError("unrelated DFlash completion did not drain")
        resume_started_ns = _now_ns()
        FlashMTPDraftEngine._poll_messages(engine)
        resume_elapsed_ns = _now_ns() - resume_started_ns
        return {
            "first_poll_ns": first_elapsed_ns,
            "event_pending_after_first_poll": pending_after_first_poll,
            "deferred_controls_after_first_poll": deferred_after_first_poll,
            "migration_status_published_while_pending": bool(
                status_count_while_pending
            ),
            "unrelated_cancel_processed_while_pending": cancel_id
            in engine._cancelled_ids,
            "scheduler_ticks_while_pending": scheduler_ticks,
            "max_control_poll_ns": max(control_poll_ns),
            "gpu_ready_ns": ready_ns,
            "resume_started_ns": resume_started_ns,
            "resume_elapsed_ns": resume_elapsed_ns,
            "deferred_controls_after_resume": len(
                engine._deferred_migration_gpu_controls
            ),
        }

    destination, destination_statuses = _build_engine("migration-destination")
    prepare_completion = _launch_unrelated_dflash(destination, "prepare")
    prepare_message = DraftKvMigrationPrepare(
        protocol_version=PROTOCOL_VERSION,
        migration_id="migration-control-liveness",
        source_draft_instance_id="migration-source",
        destination_draft_instance_id=destination.instance_id,
        target_instance_id="migration-target",
        request_id="migration-request",
        draft_kv_handle="migration-handle",
        draft_profile_id=destination.profile.profile_id,
        round_id=1,
        committed_kv_len=committed_len,
        deadline_ns=_now_ns() + 60_000_000_000,
    )
    prepare = _poll_control_while_pending(
        destination,
        prepare_completion,
        prepare_message,
        "unrelated-prepare-cancel",
    )
    prepare_status = destination_statuses[-1]
    prepare["status"] = str(prepare_status.status)
    prepare["reservation_pages"] = len(prepare_status.reservation.page_indices)

    source, source_statuses = _build_engine("migration-source")
    token_ids = tuple(range(committed_len))
    source_req = Req(
        rid="migration-request",
        origin_input_text="",
        origin_input_ids=array("q", token_ids),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=block_size),
    )
    source_req.full_untruncated_fill_ids = source_req.origin_input_ids
    source_req.logprob_start_len = -1
    source_req.set_extend_range(committed_len, committed_len)
    source_pool = source._draft_model_runner.req_to_token_pool
    source_allocator = source._draft_model_runner.token_to_kv_pool_allocator
    if source_pool.alloc([source_req]) is None:
        raise RuntimeError("source request-row allocation failed")
    source_indices = source_allocator.alloc(committed_len)
    if source_indices is None:
        raise RuntimeError("source KV-page allocation failed")
    source_req.kv = ReqKvInfo(kv_allocated_len=committed_len, swa_evicted_seqlen=0)
    source_req.kv_committed_len = committed_len
    source_req.prefix_indices = source_indices.to(torch.int64)
    source_pool.req_to_token[source_req.req_pool_idx, :committed_len] = (
        source_indices.to(torch.int32)
    )
    source._persistent_requests["migration-handle"] = PersistentDraftRequest(
        req=source_req,
        target_instance_id="migration-target",
        request_id="migration-request",
        draft_kv_handle="migration-handle",
        valid_committed_len=committed_len,
        last_round_id=1,
        proposal_base_len=committed_len,
        committed_token_ids=list(token_ids),
    )
    torch.cuda.synchronize(device)
    export_completion = _launch_unrelated_dflash(source, "export")
    export_message = DraftKvMigrationExport(
        protocol_version=PROTOCOL_VERSION,
        migration_id="migration-control-liveness",
        source_draft_instance_id=source.instance_id,
        destination_draft_instance_id=destination.instance_id,
        target_instance_id="migration-target",
        request_id="migration-request",
        draft_kv_handle="migration-handle",
        draft_profile_id=source.profile.profile_id,
        round_id=1,
        committed_kv_len=committed_len,
        deadline_ns=_now_ns() + 60_000_000_000,
        pool=source._kv_pool,
        reservation=prepare_status.reservation,
    )
    export = _poll_control_while_pending(
        source,
        export_completion,
        export_message,
        "unrelated-export-cancel",
    )
    export_status = source_statuses[-1]
    export["status"] = str(export_status.status)
    export["source_pages"] = len(export_status.transfer.page_indices)
    export["producer_event_handle_bytes"] = len(
        export_status.transfer.producer_event_handle
    )

    gate_passed = all(
        all(
            (
                phase["event_pending_after_first_poll"],
                phase["deferred_controls_after_first_poll"] == 1,
                not phase["migration_status_published_while_pending"],
                phase["unrelated_cancel_processed_while_pending"],
                phase["scheduler_ticks_while_pending"] > 0,
                phase["max_control_poll_ns"] < 10_000_000,
                phase["resume_started_ns"] >= phase["gpu_ready_ns"],
                phase["deferred_controls_after_resume"] == 0,
                phase["status"] == "ok",
            )
        )
        for phase in (prepare, export)
    ) and bool(
        prepare["reservation_pages"] == committed_len
        and export["source_pages"] == committed_len
        and export["producer_event_handle_bytes"] > 0
    )
    return {
        "actual_control_poll": "FlashMTPDraftEngine._poll_messages",
        "actual_prepare_handler": "FlashMTPDraftEngine._handle_migration_prepare",
        "actual_export_handler": "FlashMTPDraftEngine._handle_migration_export",
        "actual_request_pool": "ReqToTokenPool",
        "actual_kv_allocator": "TokenToKVPoolAllocator",
        "prepare": prepare,
        "export": export,
        "gate_passed": gate_passed,
    }


def _draft_worker(
    config: dict[str, Any], command_queue: mp.Queue, result_queue: mp.Queue
) -> None:
    transport = None
    try:
        torch.cuda.set_device(config["destination_gpu"])
        device = torch.device("cuda", config["destination_gpu"])
        launch_gate = _dflash_launch_liveness_gate(device, config["delay_cycles"])
        two_batch_gate = _dflash_two_batch_liveness_gate(device, config["delay_cycles"])
        migration_control_gate = _dflash_migration_control_liveness_gate(
            device, config["delay_cycles"]
        )
        transport = DraftFeaturePullTransport(device=device, max_workers=1)
        injected_runtime = _InjectedFeatureRuntime(transport._runtime)
        transport._runtime = injected_runtime
        properties = torch.cuda.get_device_properties(device)
        result_queue.put(
            {
                "kind": "draft_ready",
                "pid": os.getpid(),
                "started_ns": _now_ns(),
                "device_uuid": str(properties.uuid),
                "device_name": properties.name,
                "device_total_memory": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
                "launch_liveness_gate": launch_gate,
                "two_batch_liveness_gate": two_batch_gate,
                "migration_control_liveness_gate": migration_control_gate,
            }
        )
        while True:
            command = command_queue.get()
            action = command["action"]
            if action == "pull":
                descriptor = command["descriptor"]
                expected_value = int(command["value"])
                destination = torch.empty(
                    descriptor.shape,
                    dtype=torch.bfloat16,
                    device=device,
                )
                injected_runtime.configure(
                    mode=command["mode"],
                    delay_cycles=(
                        config["delay_cycles"] if command.get("delayed") else 0
                    ),
                )
                submit_started_ns = _now_ns()
                future = transport.submit(
                    source=descriptor,
                    destination=destination,
                )
                submit_return_ns = _now_ns()
                lifecycle_engine = None
                lifecycle_releases = []
                lifecycle_failures = []
                lifecycle_outbox = _RecordingOutbox()
                if descriptor.transfer_id in {
                    "feature-delayed-timeout",
                    "feature-post-enqueue-failure",
                }:
                    from sglang.srt.speculative.dt_draft_engine import (
                        FlashMTPDraftEngine,
                        PendingDraftTask,
                        PendingFeatureBatch,
                    )

                    lifecycle_engine = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
                    lifecycle_task = SimpleNamespace(
                        transfer_id=descriptor.transfer_id,
                        request_id=f"request-{descriptor.transfer_id}",
                        round_id=1,
                        target_instance_id=descriptor.target_instance_id,
                        draft_kv_handle=f"handle-{descriptor.transfer_id}",
                        source=descriptor,
                        source_read_migration_id="",
                        deadline_ns=_now_ns() + int(config["timeout_s"] * 1e9),
                    )
                    lifecycle_engine.instance_id = config["draft_instance_id"]
                    lifecycle_engine._client = SimpleNamespace(
                        instance_session_id=config["draft_session_id"],
                        coordinator_session_id=config["coordinator_session_id"],
                    )
                    lifecycle_engine._delivery_outbox = lifecycle_outbox
                    lifecycle_engine._flush_reliable_outbox = lambda: None
                    lifecycle_engine.batch_source_read_done = False
                    lifecycle_engine._source_done_pending = deque()
                    lifecycle_engine._pending_feature_batches = deque(
                        [
                            PendingFeatureBatch(
                                pending=[
                                    PendingDraftTask(
                                        task=lifecycle_task, pivot_bytes=None
                                    )
                                ],
                                staging_slot=0,
                                row_ranges=((0, int(descriptor.shape[0])),),
                                futures=(future,),
                                submitted_ns=submit_started_ns,
                            )
                        ]
                    )
                    lifecycle_engine._pending_dflash_completions = deque()
                    lifecycle_engine._cancelled_ids = set()
                    lifecycle_engine._feature_source_inflight = set()
                    lifecycle_engine._feature_source_quiescent = set()
                    lifecycle_engine._feature_source_unquiescent = set()
                    lifecycle_engine._deferred_feature_source_tasks = {}
                    lifecycle_engine._free_feature_staging_slots = deque()
                    lifecycle_engine._stop = False
                    lifecycle_engine._release_persistent_request = (
                        lambda handle: lifecycle_releases.append(handle)
                    )
                    lifecycle_engine._publish_failures = (
                        lambda failures: lifecycle_failures.extend(
                            (task.transfer_id, type(exc).__name__)
                            for task, exc in failures
                        )
                    )
                    lifecycle_engine._publish_capacity = lambda: None
                    lifecycle_engine._resume_feature_finalizations = lambda: None
                result_queue.put(
                    {
                        "kind": "pull_submitted",
                        "transfer_id": descriptor.transfer_id,
                        "submit_started_ns": submit_started_ns,
                        "submit_return_ns": submit_return_ns,
                        "future_done_at_submit_return": future.done(),
                    }
                )
                scheduler_ticks = 0
                lifecycle_poll_ticks = 0
                lifecycle_poll_max_ns = 0
                control_acks = []
                enqueued_reported = False
                deadline = time.monotonic() + config["timeout_s"]
                while not future.done():
                    enqueued, enqueued_ns = injected_runtime.enqueue_state()
                    if enqueued and not enqueued_reported:
                        result_queue.put(
                            {
                                "kind": "copy_enqueued",
                                "transfer_id": descriptor.transfer_id,
                                "enqueued_ns": enqueued_ns,
                                "future_done": future.done(),
                            }
                        )
                        enqueued_reported = True
                    try:
                        control = command_queue.get_nowait()
                    except queue.Empty:
                        control = None
                    if control is not None:
                        if control["action"] != "control":
                            raise RuntimeError(
                                f"unexpected command while pull pending: {control}"
                            )
                        ack = {
                            "kind": "control_ack",
                            "name": control["name"],
                            "transfer_id": descriptor.transfer_id,
                            "processed_ns": _now_ns(),
                            "future_done": future.done(),
                        }
                        control_acks.append(ack)
                        result_queue.put(ack)
                        if control["name"] == "cancel" and lifecycle_engine is not None:
                            lifecycle_engine._cancelled_ids.add(descriptor.transfer_id)
                    if lifecycle_engine is not None:
                        poll_started_ns = _now_ns()
                        FlashMTPDraftEngine._poll_feature_batches(lifecycle_engine)
                        lifecycle_poll_max_ns = max(
                            lifecycle_poll_max_ns, _now_ns() - poll_started_ns
                        )
                        lifecycle_poll_ticks += 1
                    scheduler_ticks += 1
                    if time.monotonic() >= deadline:
                        raise TimeoutError("feature pull gate exceeded its deadline")
                    time.sleep(config["poll_sleep_s"])
                if lifecycle_engine is not None:
                    FlashMTPDraftEngine._poll_feature_batches(lifecycle_engine)
                enqueued, enqueued_ns = injected_runtime.enqueue_state()
                if enqueued and not enqueued_reported:
                    result_queue.put(
                        {
                            "kind": "copy_enqueued",
                            "transfer_id": descriptor.transfer_id,
                            "enqueued_ns": enqueued_ns,
                            "future_done": True,
                        }
                    )
                error = ""
                metrics = None
                resources_quiescent = True
                try:
                    metrics = future.result()
                except Exception as exc:  # noqa: BLE001 - injected evidence
                    error = f"{type(exc).__name__}: {exc}"
                    resources_quiescent = bool(
                        getattr(exc, "resources_quiescent", True)
                    )
                destination_valid = bool(
                    torch.all(destination == expected_value).item()
                )
                source_read_done = None
                for operation, _message_id, message in lifecycle_outbox.entries:
                    if operation == "source_read_done":
                        source_read_done = {
                            "protocol_version": message.protocol_version,
                            "transfer_id": message.transfer_id,
                            "target_instance_id": message.target_instance_id,
                            "draft_instance_id": message.draft_instance_id,
                            "source_slot_id": message.source_slot_id,
                            "slot_generation": message.slot_generation,
                            "completed_ns": message.completed_ns,
                            "source_session_id": message.source_session_id,
                            "pool_generation": message.pool_generation,
                            "instance_session_id": message.instance_session_id,
                            "coordinator_session_id": message.coordinator_session_id,
                        }
                        break
                result_queue.put(
                    {
                        "kind": "pull_complete",
                        "transfer_id": descriptor.transfer_id,
                        "success": not error,
                        "error": error,
                        "resources_quiescent": resources_quiescent,
                        "destination_valid": destination_valid,
                        "scheduler_ticks_while_pending": scheduler_ticks,
                        "lifecycle_poll": (
                            None
                            if lifecycle_engine is None
                            else {
                                "actual_poll_method": "FlashMTPDraftEngine._poll_feature_batches",
                                "poll_ticks_while_pending": lifecycle_poll_ticks,
                                "max_poll_ns": lifecycle_poll_max_ns,
                                "cancel_applied": bool(
                                    descriptor.transfer_id == "feature-delayed-timeout"
                                    and descriptor.transfer_id
                                    not in lifecycle_engine._cancelled_ids
                                    and lifecycle_releases
                                ),
                                "released_handles": lifecycle_releases,
                                "published_failures": lifecycle_failures,
                                "source_read_done": source_read_done,
                            }
                        ),
                        "control_acks": control_acks,
                        "completed_ns": _now_ns(),
                        "metrics": (
                            None
                            if metrics is None
                            else {
                                "submitted_ns": metrics.submitted_ns,
                                "gpu_completed_ns": metrics.gpu_completed_ns,
                                "mapping_ns": metrics.mapping_ns,
                                "driver_submit_ns": metrics.driver_submit_ns,
                                "gpu_copy_ns": metrics.gpu_copy_ns,
                                "feature_bytes": metrics.feature_bytes,
                                "cold_mapping_count": metrics.cold_mapping_count,
                            }
                        ),
                    }
                )
                continue
            if action == "stop":
                transport.close()
                transport = None
                result_queue.put({"kind": "draft_stopped"})
                return
            raise ValueError(f"unknown draft command {action!r}")
    except Exception as exc:  # noqa: BLE001 - report child-process failure
        result_queue.put(
            {
                "kind": "error",
                "where": "draft",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if transport is not None:
            transport.close()


def _get(result_queue: mp.Queue, timeout_s: float) -> dict:
    item = result_queue.get(timeout=timeout_s)
    if item.get("kind") == "error":
        raise RuntimeError(
            f"{item.get('where')} child failed: {item.get('error')}\n"
            f"{item.get('traceback')}"
        )
    return item


def _source_request(
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    timeout_s: float,
    command: dict,
) -> dict:
    command_queue.put(command)
    return _get(result_queue, timeout_s)


def _publish(
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    config: dict,
    transfer_id: str,
    value: int,
) -> dict:
    result = _source_request(
        command_queue,
        result_queue,
        config["timeout_s"],
        {"action": "publish", "transfer_id": transfer_id, "value": value},
    )
    if result.get("kind") != "published":
        raise RuntimeError(f"unexpected publication response: {result}")
    return result


def _read_done(
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    config: dict,
    descriptor,
    **overrides,
) -> dict:
    message = overrides.pop("message", None)
    result = _source_request(
        command_queue,
        result_queue,
        config["timeout_s"],
        {
            "action": "read_done",
            "descriptor": descriptor,
            "message": message,
            **overrides,
        },
    )
    if result.get("kind") != "read_done_result":
        raise RuntimeError(f"unexpected read-done response: {result}")
    return result


def _run_pull(
    *,
    draft_command_queue: mp.Queue,
    draft_result_queue: mp.Queue,
    config: dict,
    descriptor,
    value: int,
    mode: str,
    delayed: bool,
    control_names: tuple[str, ...] = (),
    on_enqueued=None,
) -> dict:
    draft_command_queue.put(
        {
            "action": "pull",
            "descriptor": descriptor,
            "value": value,
            "mode": mode,
            "delayed": delayed,
        }
    )
    submitted = None
    enqueued = None
    control_acks = []
    completion = None
    controls_sent = False
    callback_result = None
    deadline = time.monotonic() + config["timeout_s"]
    while completion is None:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {descriptor.transfer_id}")
        item = _get(draft_result_queue, config["timeout_s"])
        kind = item.get("kind")
        if kind == "pull_submitted":
            submitted = item
            if not controls_sent:
                for name in control_names:
                    draft_command_queue.put({"action": "control", "name": name})
                controls_sent = True
        elif kind == "copy_enqueued":
            enqueued = item
            if on_enqueued is not None and callback_result is None:
                callback_result = on_enqueued()
        elif kind == "control_ack":
            control_acks.append(item)
        elif kind == "pull_complete":
            completion = item
        else:
            raise RuntimeError(f"unexpected Draft response: {item}")
    return {
        "submitted": submitted,
        "enqueued": enqueued,
        "control_acks": control_acks,
        "source_while_enqueued": callback_result,
        "completion": completion,
    }


def _start_source(
    ctx,
    config: dict,
    generation_index: int,
) -> tuple[mp.Process, mp.Queue, mp.Queue, dict]:
    command_queue = ctx.Queue()
    result_queue = ctx.Queue()
    source_session_id = f"feature-source-session-{generation_index}-{uuid.uuid4().hex}"
    process = ctx.Process(
        target=_source_worker,
        args=(
            config,
            source_session_id,
            generation_index,
            command_queue,
            result_queue,
        ),
        name=f"dt-feature-source-{generation_index}",
    )
    process.start()
    ready = _get(result_queue, config["timeout_s"])
    if ready.get("kind") != "source_ready":
        raise RuntimeError(f"unexpected source startup response: {ready}")
    return process, command_queue, result_queue, ready


def _stop_child(
    process: mp.Process | None,
    command_queue: mp.Queue | None,
    result_queue: mp.Queue | None,
    timeout_s: float,
    expected_kind: str,
) -> dict | None:
    if process is None:
        return None
    result = None
    if process.is_alive() and command_queue is not None and result_queue is not None:
        command_queue.put({"action": "stop"})
        try:
            result = _get(result_queue, timeout_s)
            if result.get("kind") != expected_kind:
                raise RuntimeError(f"unexpected shutdown response: {result}")
        except queue.Empty:
            result = None
    process.join(timeout=min(timeout_s, 10))
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    return result


def run(config: dict[str, Any]) -> dict:
    ctx = mp.get_context("spawn")
    draft_commands = ctx.Queue()
    draft_results = ctx.Queue()
    draft = ctx.Process(
        target=_draft_worker,
        args=(config, draft_commands, draft_results),
        name="dt-feature-draft",
    )
    draft.start()
    source = None
    source_commands = None
    source_results = None
    replacement = None
    replacement_commands = None
    replacement_results = None
    try:
        draft_ready = _get(draft_results, config["timeout_s"])
        if draft_ready.get("kind") != "draft_ready":
            raise RuntimeError(f"unexpected Draft startup response: {draft_ready}")
        source, source_commands, source_results, source_ready = _start_source(
            ctx, config, 0
        )

        delayed_publish = _publish(
            source_commands, source_results, config, "feature-delayed-timeout", 11
        )
        delayed_descriptor = delayed_publish["descriptor"]

        def _expire_and_probe():
            expired = _source_request(
                source_commands,
                source_results,
                config["timeout_s"],
                {
                    "action": "expire",
                    "transfer_id": delayed_descriptor.transfer_id,
                },
            )
            probe = _source_request(
                source_commands,
                source_results,
                config["timeout_s"],
                {
                    "action": "probe_publish",
                    "transfer_id": "probe-before-delayed-quiescence",
                    "value": 99,
                },
            )
            return {"expired": expired, "reuse_probe": probe}

        delayed_pull = _run_pull(
            draft_command_queue=draft_commands,
            draft_result_queue=draft_results,
            config=config,
            descriptor=delayed_descriptor,
            value=11,
            mode="normal",
            delayed=True,
            control_names=("cancel", "timeout", "unrelated-health"),
            on_enqueued=_expire_and_probe,
        )
        delayed_read_done = _read_done(
            source_commands,
            source_results,
            config,
            delayed_descriptor,
            message=delayed_pull["completion"]["lifecycle_poll"]["source_read_done"],
        )
        delayed_read_done_retry = _read_done(
            source_commands,
            source_results,
            config,
            delayed_descriptor,
            message=delayed_pull["completion"]["lifecycle_poll"]["source_read_done"],
        )

        failure_publish = _publish(
            source_commands, source_results, config, "feature-post-enqueue-failure", 22
        )
        failure_descriptor = failure_publish["descriptor"]

        def _probe_failure_lease():
            return _source_request(
                source_commands,
                source_results,
                config["timeout_s"],
                {
                    "action": "probe_publish",
                    "transfer_id": "probe-before-failure-quiescence",
                    "value": 98,
                },
            )

        failure_pull = _run_pull(
            draft_command_queue=draft_commands,
            draft_result_queue=draft_results,
            config=config,
            descriptor=failure_descriptor,
            value=22,
            mode="fail_after_enqueue",
            delayed=True,
            control_names=("unrelated-failure-health",),
            on_enqueued=_probe_failure_lease,
        )
        failure_read_done = _read_done(
            source_commands,
            source_results,
            config,
            failure_descriptor,
            message=failure_pull["completion"]["lifecycle_poll"]["source_read_done"],
        )
        failure_read_done_retry = _read_done(
            source_commands,
            source_results,
            config,
            failure_descriptor,
            message=failure_pull["completion"]["lifecycle_poll"]["source_read_done"],
        )

        stale_publish = _publish(
            source_commands, source_results, config, "feature-stale-generation", 33
        )
        stale_descriptor = stale_publish["descriptor"]
        old_read_done_after_reuse = _read_done(
            source_commands, source_results, config, delayed_descriptor
        )
        wrong_generation = _read_done(
            source_commands,
            source_results,
            config,
            stale_descriptor,
            slot_generation=stale_descriptor.slot_generation - 1,
        )
        stale_session = _read_done(
            source_commands,
            source_results,
            config,
            stale_descriptor,
            source_session_id="retired-feature-source-session",
        )
        stale_status = _source_request(
            source_commands,
            source_results,
            config["timeout_s"],
            {"action": "status"},
        )
        stale_pull = _run_pull(
            draft_command_queue=draft_commands,
            draft_result_queue=draft_results,
            config=config,
            descriptor=stale_descriptor,
            value=33,
            mode="normal",
            delayed=False,
        )
        stale_read_done = _read_done(
            source_commands, source_results, config, stale_descriptor
        )

        reuse_publish = _publish(
            source_commands, source_results, config, "feature-safe-reuse", 44
        )
        reuse_descriptor = reuse_publish["descriptor"]
        reuse_pull = _run_pull(
            draft_command_queue=draft_commands,
            draft_result_queue=draft_results,
            config=config,
            descriptor=reuse_descriptor,
            value=44,
            mode="normal",
            delayed=False,
        )
        reuse_read_done = _read_done(
            source_commands, source_results, config, reuse_descriptor
        )
        source_stop = _stop_child(
            source,
            source_commands,
            source_results,
            config["timeout_s"],
            "source_stopped",
        )
        source = None

        (
            replacement,
            replacement_commands,
            replacement_results,
            replacement_ready,
        ) = _start_source(ctx, config, 1)
        replacement_publish = _publish(
            replacement_commands,
            replacement_results,
            config,
            "feature-process-replacement",
            55,
        )
        replacement_descriptor = replacement_publish["descriptor"]
        replacement_pull = _run_pull(
            draft_command_queue=draft_commands,
            draft_result_queue=draft_results,
            config=config,
            descriptor=replacement_descriptor,
            value=55,
            mode="normal",
            delayed=False,
        )
        replacement_read_done = _read_done(
            replacement_commands,
            replacement_results,
            config,
            replacement_descriptor,
        )
        replacement_read_done_retry = _read_done(
            replacement_commands,
            replacement_results,
            config,
            replacement_descriptor,
        )
        replacement_stop = _stop_child(
            replacement,
            replacement_commands,
            replacement_results,
            config["timeout_s"],
            "source_stopped",
        )
        replacement = None
        draft_stop = _stop_child(
            draft,
            draft_commands,
            draft_results,
            config["timeout_s"],
            "draft_stopped",
        )

        delayed_controls = {item["name"]: item for item in delayed_pull["control_acks"]}
        failure_controls = {item["name"]: item for item in failure_pull["control_acks"]}
        checks = {
            "actual_feature_classes": True,
            "separate_processes_and_devices": bool(
                source_ready["pid"] != draft_ready["pid"]
                and config["source_gpu"] != config["destination_gpu"]
            ),
            "launch_returns_with_gpu_pending": draft_ready["launch_liveness_gate"][
                "gate_passed"
            ],
            "successive_batch_submission_nonblocking": draft_ready[
                "two_batch_liveness_gate"
            ]["gate_passed"],
            "migration_prepare_export_controls_nonblocking": draft_ready[
                "migration_control_liveness_gate"
            ]["gate_passed"],
            "delayed_pull_nonblocking": bool(
                delayed_pull["submitted"]
                and not delayed_pull["submitted"]["future_done_at_submit_return"]
                and delayed_pull["completion"]["scheduler_ticks_while_pending"] > 0
            ),
            "cancel_timeout_and_unrelated_control_progress": bool(
                set(delayed_controls) == {"cancel", "timeout", "unrelated-health"}
                and all(not item["future_done"] for item in delayed_controls.values())
                and delayed_pull["completion"]["lifecycle_poll"]["cancel_applied"]
                and delayed_pull["completion"]["lifecycle_poll"]["source_read_done"]
                is not None
            ),
            "timeout_quarantines_without_reuse": bool(
                delayed_pull["source_while_enqueued"]["expired"]["quarantined_slots"]
                == 1
                and delayed_pull["source_while_enqueued"]["reuse_probe"]["blocked"]
                and delayed_pull["source_while_enqueued"]["reuse_probe"]["free_slots"]
                == 0
            ),
            "delayed_copy_and_read_done_retry": bool(
                delayed_pull["completion"]["success"]
                and delayed_pull["completion"]["destination_valid"]
                and delayed_read_done["applied"]
                and delayed_read_done["free_slots"] == 1
                and delayed_read_done_retry["applied"]
                and delayed_read_done_retry["free_slots"] == 1
                and delayed_read_done_retry["ack_outbox_entries"]
                == delayed_read_done["ack_outbox_entries"]
            ),
            "post_enqueue_failure_quiesced_before_reuse": bool(
                failure_pull["enqueued"]
                and not failure_pull["enqueued"]["future_done"]
                and failure_pull["source_while_enqueued"]["blocked"]
                and not failure_pull["completion"]["success"]
                and "injected post-enqueue" in failure_pull["completion"]["error"]
                and failure_pull["completion"]["resources_quiescent"]
                and failure_pull["completion"]["destination_valid"]
                and failure_pull["completion"]["lifecycle_poll"]["published_failures"]
                == [(failure_descriptor.transfer_id, "RuntimeError")]
                and failure_pull["completion"]["lifecycle_poll"]["source_read_done"]
                is not None
                and failure_controls["unrelated-failure-health"]["future_done"] is False
                and failure_read_done["applied"]
                and failure_read_done_retry["applied"]
                and failure_read_done_retry["ack_outbox_entries"]
                == failure_read_done["ack_outbox_entries"]
                and failure_publish["published_ns"] > delayed_read_done["observed_ns"]
            ),
            "stale_fences_do_not_release_current_slot": bool(
                old_read_done_after_reuse["applied"]
                and old_read_done_after_reuse["free_slots"] == 0
                and "does not match" in wrong_generation["error"]
                and "stale source session" in stale_session["error"]
                and stale_status["free_slots"] == 0
                and stale_status["active_transfers"] == [stale_descriptor.transfer_id]
            ),
            "safe_reuse_after_quiescence": bool(
                stale_pull["completion"]["success"]
                and stale_pull["completion"]["destination_valid"]
                and stale_read_done["free_slots"] == 1
                and reuse_descriptor.slot_generation
                == stale_descriptor.slot_generation + 1
                and reuse_publish["published_ns"] > stale_read_done["observed_ns"]
                and reuse_pull["completion"]["success"]
                and reuse_pull["completion"]["destination_valid"]
                and reuse_read_done["free_slots"] == 1
            ),
            "process_replacement_fenced_and_remapped": bool(
                replacement_ready["source_session_id"]
                != source_ready["source_session_id"]
                and replacement_ready["pool_generation"]
                != source_ready["pool_generation"]
                and replacement_pull["completion"]["success"]
                and replacement_pull["completion"]["destination_valid"]
                and replacement_pull["completion"]["metrics"]["cold_mapping_count"] == 1
                and replacement_read_done["free_slots"] == 1
                and replacement_read_done_retry["free_slots"] == 1
                and replacement_read_done_retry["ack_outbox_entries"]
                == replacement_read_done["ack_outbox_entries"]
                and replacement_ready["started_ns"] > source_stop["stopped_ns"]
            ),
        }
        return {
            "schema_version": 1,
            "benchmark": "dt-dflash-feature-p2p-lifetime",
            "host": socket.gethostname(),
            "timestamp_ns": _now_ns(),
            "config": config,
            "implementation": {
                "source_pool": "sglang.srt.speculative.dt_feature_p2p.TargetFeatureSourcePool",
                "draft_transport": "sglang.srt.speculative.dt_feature_p2p.DraftFeaturePullTransport",
                "read_done_handler": "sglang.srt.speculative.dt_target.DTTargetController._consume_source_read_done",
                "feature_batch_poll": "sglang.srt.speculative.dt_draft_engine.FlashMTPDraftEngine._poll_feature_batches",
                "completion_poll": "sglang.srt.speculative.dt_draft_engine.FlashMTPDraftEngine._poll_dflash_completions",
                "migration_control_poll": "sglang.srt.speculative.dt_draft_engine.FlashMTPDraftEngine._poll_messages",
            },
            "source": source_ready,
            "draft": draft_ready,
            "delayed_cancel_timeout": {
                "publication": {
                    "slot_generation": delayed_descriptor.slot_generation,
                    "published_ns": delayed_publish["published_ns"],
                },
                "pull": delayed_pull,
                "read_done": delayed_read_done,
                "read_done_retry": delayed_read_done_retry,
            },
            "post_enqueue_failure": {
                "publication": {
                    "slot_generation": failure_descriptor.slot_generation,
                    "published_ns": failure_publish["published_ns"],
                },
                "pull": failure_pull,
                "read_done": failure_read_done,
                "read_done_retry": failure_read_done_retry,
            },
            "stale_generation": {
                "current_slot_generation": stale_descriptor.slot_generation,
                "published_ns": stale_publish["published_ns"],
                "old_read_done_after_reuse": old_read_done_after_reuse,
                "wrong_generation": wrong_generation,
                "stale_session": stale_session,
                "status_before_current_read": stale_status,
                "pull": stale_pull,
                "read_done": stale_read_done,
            },
            "safe_reuse": {
                "slot_generation": reuse_descriptor.slot_generation,
                "published_ns": reuse_publish["published_ns"],
                "pull": reuse_pull,
                "read_done": reuse_read_done,
            },
            "process_replacement": {
                "source": replacement_ready,
                "slot_generation": replacement_descriptor.slot_generation,
                "published_ns": replacement_publish["published_ns"],
                "pull": replacement_pull,
                "read_done": replacement_read_done,
                "read_done_retry": replacement_read_done_retry,
            },
            "shutdown": {
                "source": source_stop,
                "replacement_source": replacement_stop,
                "draft": draft_stop,
            },
            "checks": checks,
            "gate_passed": all(checks.values()),
        }
    finally:
        for process, commands, results, expected in (
            (source, source_commands, source_results, "source_stopped"),
            (
                replacement,
                replacement_commands,
                replacement_results,
                "source_stopped",
            ),
            (draft, draft_commands, draft_results, "draft_stopped"),
        ):
            if process is not None and process.is_alive():
                try:
                    _stop_child(
                        process,
                        commands,
                        results,
                        min(config["timeout_s"], 5),
                        expected,
                    )
                except Exception:
                    process.terminate()
                    process.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gpu", type=int, required=True)
    parser.add_argument("--destination-gpu", type=int, required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--feature-width", type=int, default=1024)
    parser.add_argument("--delay-ms", type=int, default=250)
    parser.add_argument("--poll-sleep-us", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source_gpu == args.destination_gpu:
        parser.error("source and destination GPUs must differ")
    if (
        min(
            args.rows,
            args.feature_width,
            args.delay_ms,
            args.poll_sleep_us,
        )
        <= 0
    ):
        parser.error("rows, feature width, delay, and poll sleep must be positive")
    config = {
        "source_gpu": args.source_gpu,
        "destination_gpu": args.destination_gpu,
        "rows": args.rows,
        "feature_width": args.feature_width,
        "delay_ms": args.delay_ms,
        "delay_cycles": 1_500_000 * args.delay_ms,
        "poll_sleep_s": args.poll_sleep_us / 1_000_000,
        "timeout_s": args.timeout_s,
        "target_instance_id": "feature-target-0",
        "draft_instance_id": "feature-draft-0",
        "draft_session_id": f"feature-draft-session-{uuid.uuid4().hex}",
        "coordinator_session_id": f"feature-coordinator-{uuid.uuid4().hex}",
        "feature_schema_id": "dflash-feature-gate-v1",
    }
    result = run(config)
    result["command"] = [sys.executable, *sys.argv]
    result["environment"] = {
        "python": sys.version,
        "sys_executable": sys.executable,
        "torch": torch.__version__,
        "built_cuda": torch.version.cuda,
        "cuda_runtime_library": os.environ.get("DT_CUDA_RUNTIME_LIBRARY", ""),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
