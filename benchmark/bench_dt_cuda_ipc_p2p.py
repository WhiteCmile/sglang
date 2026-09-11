"""Two-process CUDA-IPC P2P feasibility benchmark for DT KV migration.

This intentionally uses the same allocation-base registration, generation-fenced
mapping cache, coalesced page runs, and ``cudaMemcpyBatchAsync`` submission used
by the DT migration transport.  It has no host payload path: source and
destination pools remain CUDA allocations for the whole experiment.

Example (short gate on two idle GPUs)::

    python sglang/benchmark/bench_dt_cuda_ipc_p2p.py \
      --source-gpu 6 --destination-gpu 7 --sizes-mib 4 --repetitions 1

The script runs two independent source processes against one destination process.
The second process has a new pool generation, which exercises stale mapping
invalidation even if CUDA reuses a virtual address.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Let the benchmark run directly from an SGLang checkout without installation.
_SGLANG_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(_SGLANG_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SGLANG_PYTHON))

import torch  # noqa: E402
from sglang.srt.speculative.dt_cuda_ipc import (  # noqa: E402
    CudaIpcMappingCache,
    CudaIpcRuntime,
    build_copy_segments,
    coalesce_page_runs,
    register_buffers,
)
from sglang.srt.speculative.dt_kv_p2p import DraftKvP2pTransport  # noqa: E402

MIB = 1024 * 1024


def _mono_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def _parse_mib_list(value: str) -> list[int]:
    sizes = [int(part) for part in value.split(",") if part.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("--sizes-mib must be positive integers")
    return sizes


def _pool_buffers(pool: list[torch.Tensor], page_bytes: int):
    return [
        (int(buffer.data_ptr()), int(buffer.numel()), page_bytes) for buffer in pool
    ]


def _allocate_pool(config: dict[str, Any], gpu: int) -> list[torch.Tensor]:
    """Allocate one independent CUDA allocation per KV buffer/layer."""
    return [
        torch.empty(
            (config["pool_pages"], config["page_bytes"]),
            dtype=torch.uint8,
            device=f"cuda:{gpu}",
        )
        for _ in range(config["layers"])
    ]


def _pattern(
    *,
    generation_index: int,
    page_indices: list[int] | tuple[int, ...],
    layers: int,
    layer_base: int = 0,
    page_bytes: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate deterministic per-buffer/layer/page bytes entirely on GPU."""
    layer = (layer_base + torch.arange(layers, device=device, dtype=torch.int64)).view(
        layers, 1, 1
    )
    page = torch.as_tensor(page_indices, device=device, dtype=torch.int64).view(
        1, -1, 1
    )
    byte = torch.arange(page_bytes, device=device, dtype=torch.int64).view(1, 1, -1)
    return ((generation_index * 97 + layer * 31 + page * 7 + byte) % 251).to(
        torch.uint8
    )


def _source_worker(
    config: dict[str, Any],
    generation: str,
    generation_index: int,
    metadata_queue: mp.Queue,
    command_queue: mp.Queue,
) -> None:
    """Own and publish one VRAM KV pool until the receiver releases it."""
    torch.cuda.set_device(config["source_gpu"])
    runtime = CudaIpcRuntime()
    pool = _allocate_pool(config, config["source_gpu"])
    for layer_index, buffer in enumerate(pool):
        for page_start in range(0, config["pool_pages"], config["pattern_chunk_pages"]):
            page_end = min(
                page_start + config["pattern_chunk_pages"], config["pool_pages"]
            )
            buffer[page_start:page_end].copy_(
                _pattern(
                    generation_index=generation_index,
                    page_indices=tuple(range(page_start, page_end)),
                    layers=1,
                    layer_base=layer_index,
                    page_bytes=config["page_bytes"],
                    device=buffer.device,
                )[0]
            )
    # Source producer completion is made visible before publishing metadata.
    torch.cuda.synchronize(config["source_gpu"])
    producer_event = torch.cuda.Event(interprocess=True)
    producer_event.record(torch.cuda.current_stream(config["source_gpu"]))

    registration_start = _mono_ns()
    allocations, buffers = register_buffers(
        runtime, _pool_buffers(pool, config["page_bytes"])
    )
    if len(buffers) != config["layers"]:
        raise RuntimeError("source KV buffer registration count is incorrect")
    registration_us = (_mono_ns() - registration_start) / 1_000
    metadata_queue.put(
        {
            "kind": "source_registration",
            "generation": generation,
            "peer_instance_id": config["peer_instance_id"],
            "peer_session_id": f"source-session-{generation}",
            "allocations": allocations,
            "buffers": buffers,
            "registration_us": registration_us,
            "allocation_count": len(allocations),
            "buffer_count": len(buffers),
            "base_addresses": [allocation.base_address for allocation in allocations],
            "generation_index": generation_index,
            "producer_event_handle": producer_event.ipc_handle(),
        }
    )

    # Do not free or recycle the source pool until the destination has proved
    # that every possible remote read is quiescent.
    while True:
        command = command_queue.get()
        if command.get("generation") not in (generation, "*"):
            continue
        action = command.get("action")
        if action == "verify_release_deferred":
            indices = command["page_indices"]
            actual = torch.stack([buffer[indices, :] for buffer in pool], dim=0)
            expected = _pattern(
                generation_index=generation_index,
                page_indices=indices,
                layers=len(pool),
                page_bytes=config["page_bytes"],
                device=pool[0].device,
            )
            metadata_queue.put(
                {
                    "kind": "source_release_deferred",
                    "generation": generation,
                    "source_pattern_still_valid": bool(torch.equal(actual, expected)),
                    "observed_ns": _mono_ns(),
                }
            )
            continue
        if action == "release_source_lease":
            indices = command["page_indices"]
            for buffer in pool:
                buffer[indices] = command["reuse_value"]
            torch.cuda.synchronize(config["source_gpu"])
            metadata_queue.put(
                {
                    "kind": "source_pages_reused",
                    "generation": generation,
                    "reuse_value": command["reuse_value"],
                    "observed_ns": _mono_ns(),
                }
            )
            continue
        if action in {"release", "abort"}:
            return


def _page_indices(
    pool_pages: int,
    page_count: int,
    fragmented: bool,
    fragment_run_pages: int,
) -> tuple[list[int], list[int]]:
    if page_count <= 0:
        raise ValueError("page_count must be positive")
    if fragment_run_pages <= 0:
        raise ValueError("fragment run pages must be positive")
    if fragmented:
        run_pages = min(fragment_run_pages, max(1, math.ceil(page_count / 8)))
        source = [index + index // run_pages for index in range(page_count)]
    else:
        source = list(range(page_count))
    if source[-1] >= pool_pages:
        raise ValueError("pool does not have enough pages for this transfer")
    # Source and destination use the same holes in their separate pools, so
    # each corresponding run remains coalescible while run boundaries do not.
    destination = list(source)
    return source, destination


class _BoundedTransferWorker:
    """Bounded submission queue used to model the production copy worker."""

    def __init__(self, queue_depth: int) -> None:
        if queue_depth <= 0:
            raise ValueError("worker queue depth must be positive")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dt-ipc")
        self._slots = threading.BoundedSemaphore(queue_depth)

    def submit(self, function, *args, **kwargs) -> tuple[Future, int]:
        wait_start = _mono_ns()
        self._slots.acquire()
        wait_ns = _mono_ns() - wait_start
        future = self._executor.submit(function, *args, **kwargs)
        future.add_done_callback(lambda _: self._slots.release())
        return future, wait_ns

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


def _copy_once(
    *,
    runtime: CudaIpcRuntime,
    copy_stream: torch.cuda.Stream,
    destination_pool: list[torch.Tensor],
    source_bases: list[int],
    source_buffers: Any,
    destination_buffers: Any,
    source_indices: list[int],
    destination_indices: list[int],
    generation_index: int,
) -> dict[str, Any]:
    """Run the one copy primitive used by both synchronous and worker modes."""
    torch.cuda.set_device(destination_pool[0].device)
    segments = build_copy_segments(
        mapped_source_bases=source_bases,
        source_buffers=source_buffers,
        destination_buffers=destination_buffers,
        source_indices=source_indices,
        destination_indices=destination_indices,
    )
    start = torch.cuda.Event(enable_timing=True)
    complete = torch.cuda.Event(enable_timing=True)
    start_record_start = _mono_ns()
    start.record(copy_stream)
    start_record_done = _mono_ns()
    host_submit_start = _mono_ns()
    runtime.copy_batch_async(segments, int(copy_stream.cuda_stream))
    host_submit_done = _mono_ns()
    complete_record_start = _mono_ns()
    complete.record(copy_stream)
    complete_record_done = _mono_ns()

    completion_start = _mono_ns()
    sync_start = _mono_ns()
    complete.synchronize()
    sync_done = _mono_ns()
    completion_us = (sync_done - completion_start) / 1_000
    gpu_copy_us = start.elapsed_time(complete) * 1_000

    expected = _pattern(
        generation_index=generation_index,
        page_indices=source_indices,
        layers=len(destination_pool),
        page_bytes=destination_pool[0].shape[-1],
        device=destination_pool[0].device,
    )
    copied = torch.stack(
        [buffer[destination_indices, :] for buffer in destination_pool], dim=0
    )
    if not bool(torch.equal(copied, expected)):
        raise RuntimeError("destination GPU KV validation failed")

    byte_count = sum(segment.length for segment in segments)
    bandwidth_gbps = (
        (byte_count / (gpu_copy_us / 1_000_000)) / 1e9 if gpu_copy_us else 0.0
    )
    return {
        "bytes": byte_count,
        "segments": len(segments),
        "coalesced_page_runs": len(
            coalesce_page_runs(source_indices, destination_indices)
        ),
        "host_submit_us": (host_submit_done - host_submit_start) / 1_000,
        "completion_wait_us": completion_us,
        "gpu_copy_us": gpu_copy_us,
        "bandwidth_GBps": bandwidth_gbps,
        "scheduler_ticks_while_future_in_flight": 0,
        "copy_start_event_record_host_ns": {
            "start": start_record_start,
            "done": start_record_done,
        },
        "copy_complete_event_record_host_ns": {
            "start": complete_record_start,
            "done": complete_record_done,
        },
        "copy_complete_event_sync_host_ns": {
            "start": sync_start,
            "done": sync_done,
        },
    }


def _start_compute_trial(
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    trial_id: str,
    iterations: int,
) -> dict[str, Any]:
    command_queue.put({"action": "run", "trial_id": trial_id, "iterations": iterations})
    try:
        started = result_queue.get(timeout=120)
        if (
            started.get("kind") != "compute_started"
            or started.get("trial_id") != trial_id
        ):
            raise RuntimeError(f"unexpected compute start message: {started}")
    except queue.Empty as exc:
        raise RuntimeError("timed out waiting for fixed-work compute start") from exc
    return started


def _finish_compute_trial(
    result_queue: mp.Queue,
    trial_id: str,
) -> dict[str, Any]:
    try:
        completed = result_queue.get(timeout=120)
    except queue.Empty as exc:
        raise RuntimeError(
            "timed out waiting for fixed-work compute completion"
        ) from exc
    if completed.get("kind") != "compute_done" or completed.get("trial_id") != trial_id:
        raise RuntimeError(f"unexpected compute completion message: {completed}")
    return completed


def _run_compute_only_trial(
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    trial_id: str,
    iterations: int,
) -> dict[str, Any]:
    started = _start_compute_trial(command_queue, result_queue, trial_id, iterations)
    completed = _finish_compute_trial(result_queue, trial_id)
    return {**started, **completed}


def _wait_worker_future(future: Future, poll_sleep_us: int) -> tuple[Any, int]:
    scheduler_ticks = 0
    while not future.done():
        scheduler_ticks += 1
        if poll_sleep_us:
            time.sleep(poll_sleep_us / 1_000_000)
        else:
            time.sleep(0)
    return future.result(), scheduler_ticks


def _post_enqueue_failure_gate(
    *,
    runtime: CudaIpcRuntime,
    destination_pool: list[torch.Tensor],
    destination_buffers: Any,
    source: dict[str, Any],
    mapped_bases: list[int],
) -> dict[str, Any]:
    """Exercise the production worker after a real enqueue, then reuse pages."""
    page_count = 3
    source_indices, destination_indices = _page_indices(
        source["pool_pages"],
        page_count,
        fragmented=True,
        fragment_run_pages=source["fragment_run_pages"],
    )
    source_pool = SimpleNamespace(
        instance_session_id=source["peer_session_id"],
        pool_generation=source["generation"],
        allocations=source["allocations"],
        buffers=tuple(
            SimpleNamespace(
                allocation_index=buffer.allocation_index,
                allocation_offset=buffer.allocation_offset,
                visible_length=buffer.length,
                item_length=buffer.item_length,
            )
            for buffer in source["buffers"]
        ),
    )
    source_transfer = SimpleNamespace(
        page_indices=source_indices,
        token_ids=tuple(range(page_count)),
        producer_event_handle=source["producer_event_handle"],
        kv_bytes=sum(buffer.item_length for buffer in source["buffers"]) * page_count,
    )
    reservation = SimpleNamespace(page_indices=destination_indices)

    mapped_by_handle = {
        allocation.ipc_handle: mapped
        for allocation, mapped in zip(source["allocations"], mapped_bases, strict=True)
    }

    class ExistingMappingCache:
        @staticmethod
        def map(*, ipc_handle, **_identity):
            return mapped_by_handle[ipc_handle], True

        @staticmethod
        def close() -> None:
            return None

    transport = DraftKvP2pTransport.__new__(DraftKvP2pTransport)
    transport.device_index = int(destination_pool[0].device.index)
    transport.device = destination_pool[0].device
    transport._runtime = runtime
    transport._mapping_cache = ExistingMappingCache()
    transport._buffers = destination_buffers
    transport._executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="dt-gate"
    )
    transport._copy_stream = torch.cuda.Stream(device=transport.device)
    transport._fatal_error = None
    transport._closed = False
    real_copy = runtime.copy_batch_async
    terminal = torch.cuda.Event()
    injected = {"enqueued": False}

    def fail_after_enqueue(segments, stream):
        real_copy(segments, stream)
        terminal.record(transport._copy_stream)
        injected["enqueued"] = True
        raise RuntimeError("injected post-enqueue failure")

    runtime.copy_batch_async = fail_after_enqueue
    future = transport.submit(
        source_draft_instance_id=source["peer_instance_id"],
        source_pool=source_pool,
        transfer=source_transfer,
        reservation=reservation,
    )
    try:
        future.result(timeout=120)
        future_error = None
    except Exception as exc:  # noqa: BLE001 - this is the injected failure
        future_error = repr(exc)
    future_done_ns = _mono_ns()
    event_ready_before_future = terminal.query()
    terminal_sync_start = _mono_ns()
    terminal.synchronize()
    terminal_sync_done = _mono_ns()
    copied = torch.stack(
        [buffer[destination_indices, :] for buffer in destination_pool], dim=0
    )
    expected = _pattern(
        generation_index=source["generation_index"],
        page_indices=source_indices,
        layers=len(destination_pool),
        page_bytes=destination_pool[0].shape[-1],
        device=destination_pool[0].device,
    )
    copied_pattern_valid = bool(torch.equal(copied, expected))
    for buffer in destination_pool:
        buffer[destination_indices] = 173
    torch.cuda.synchronize(destination_pool[0].device)
    reuse_ok = all(
        bool(torch.all(buffer[destination_indices] == 173).item())
        for buffer in destination_pool
    )
    runtime.copy_batch_async = real_copy
    transport.close()
    return {
        "future_error": future_error,
        "enqueue_observed": injected["enqueued"],
        "future_completed_after_terminal_event": event_ready_before_future,
        "copied_pattern_valid": copied_pattern_valid,
        "terminal_event_sync_us": (terminal_sync_done - terminal_sync_start) / 1_000,
        "future_done_monotonic_ns": future_done_ns,
        "destination_reuse_stable": reuse_ok,
        "gate_passed": bool(
            injected["enqueued"]
            and future_error
            and event_ready_before_future
            and copied_pattern_valid
            and reuse_ok
        ),
    }


def _source_reuse_lifetime_gate(
    *,
    runtime: CudaIpcRuntime,
    destination_pool: list[torch.Tensor],
    destination_buffers: Any,
    source: dict[str, Any],
    mapped_bases: list[int],
    command_queue: mp.Queue,
    metadata_queue: mp.Queue,
    delay_ms: int,
) -> dict[str, Any]:
    """Cancel after enqueue and prove source reuse follows the copy fence."""
    page_count = 3
    source_indices, destination_indices = _page_indices(
        source["pool_pages"],
        page_count,
        fragmented=True,
        fragment_run_pages=source["fragment_run_pages"],
    )
    source_buffers = tuple(
        SimpleNamespace(
            allocation_index=buffer.allocation_index,
            allocation_offset=buffer.allocation_offset,
            address=0,
            length=buffer.length,
            item_length=buffer.item_length,
        )
        for buffer in source["buffers"]
    )
    segments = build_copy_segments(
        mapped_source_bases=mapped_bases,
        source_buffers=source_buffers,
        destination_buffers=destination_buffers,
        source_indices=source_indices,
        destination_indices=destination_indices,
    )
    copy_stream = torch.cuda.Stream(device=destination_pool[0].device)
    source_ready = torch.cuda.Event.from_ipc_handle(
        int(destination_pool[0].device.index), source["producer_event_handle"]
    )
    enqueued = threading.Event()
    real_copy = runtime.copy_batch_async

    class DelayedRuntime:
        @staticmethod
        def copy_batch_async(copy_segments, stream) -> None:
            # H800 clocks vary under power management; an approximate delay
            # is sufficient because the gate observes enqueue/completion
            # events rather than inferring ordering from the requested delay.
            cycles = max(1, int(1_500_000 * delay_ms))
            torch.cuda._sleep(cycles)
            real_copy(copy_segments, stream)
            enqueued.set()

    transport = DraftKvP2pTransport.__new__(DraftKvP2pTransport)
    transport.device = destination_pool[0].device
    transport._runtime = DelayedRuntime()
    transport._copy_stream = copy_stream
    transport._fatal_error = None
    cancellation_ns = 0
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="dt-lifetime-gate") as ex:
        future = ex.submit(
            transport._copy_and_wait,
            segments=segments,
            source_ready=source_ready,
            mapping_started_ns=_mono_ns(),
            mapping_done_ns=_mono_ns(),
            kv_bytes=sum(segment.length for segment in segments),
            page_count=page_count,
            cold_mapping_count=0,
        )
        if not enqueued.wait(timeout=30):
            if future.done():
                future.result()
            raise RuntimeError("lifetime gate did not observe CUDA copy enqueue")
        cancellation_ns = _mono_ns()
        future_done_at_cancellation = future.done()
        command_queue.put(
            {
                "action": "verify_release_deferred",
                "generation": source["generation"],
                "page_indices": source_indices,
            }
        )
        deferred = metadata_queue.get(timeout=30)
        if deferred.get("kind") != "source_release_deferred":
            raise RuntimeError(f"unexpected source lifetime response: {deferred}")
        future_done_before_source_reuse = future.done()
        future.result(timeout=120)
        copy_quiesced_ns = _mono_ns()

    copied = torch.stack(
        [buffer[destination_indices, :] for buffer in destination_pool], dim=0
    )
    expected = _pattern(
        generation_index=source["generation_index"],
        page_indices=source_indices,
        layers=len(destination_pool),
        page_bytes=destination_pool[0].shape[-1],
        device=destination_pool[0].device,
    )
    copied_before_reuse_valid = bool(torch.equal(copied, expected))

    reuse_value = 173
    command_queue.put(
        {
            "action": "release_source_lease",
            "generation": source["generation"],
            "page_indices": source_indices,
            "reuse_value": reuse_value,
        }
    )
    reused = metadata_queue.get(timeout=30)
    if reused.get("kind") != "source_pages_reused":
        raise RuntimeError(f"unexpected source reuse response: {reused}")
    with torch.cuda.stream(copy_stream):
        real_copy(segments, int(copy_stream.cuda_stream))
    copy_stream.synchronize()
    copied_after_reuse = torch.stack(
        [buffer[destination_indices, :] for buffer in destination_pool], dim=0
    )
    source_reuse_visible = bool(torch.all(copied_after_reuse == reuse_value).item())
    ordering_valid = bool(
        deferred["observed_ns"] >= cancellation_ns
        and reused["observed_ns"] >= copy_quiesced_ns
    )
    return {
        "cancellation_after_enqueue_ns": cancellation_ns,
        "source_release_deferred_ns": deferred["observed_ns"],
        "destination_copy_quiesced_ns": copy_quiesced_ns,
        "source_pages_reused_ns": reused["observed_ns"],
        "future_done_at_deferred_release": future_done_before_source_reuse,
        "future_done_at_cancellation": future_done_at_cancellation,
        "source_pattern_pinned_before_quiescence": deferred[
            "source_pattern_still_valid"
        ],
        "destination_copy_valid_before_source_reuse": copied_before_reuse_valid,
        "source_reuse_visible_after_quiescence": source_reuse_visible,
        "ordering_valid": ordering_valid,
        "gate_passed": bool(
            not future_done_at_cancellation
            and deferred["source_pattern_still_valid"]
            and copied_before_reuse_valid
            and source_reuse_visible
            and ordering_valid
        ),
    }


def _destination_worker(
    config: dict[str, Any],
    metadata_queue: mp.Queue,
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    compute_command_queue: mp.Queue | None,
    compute_result_queue: mp.Queue | None,
) -> None:
    cache: CudaIpcMappingCache | None = None
    transfer_worker: _BoundedTransferWorker | None = None
    try:
        torch.cuda.set_device(config["destination_gpu"])
        runtime = CudaIpcRuntime()
        destination_pool = _allocate_pool(config, config["destination_gpu"])
        _, destination_buffers = register_buffers(
            runtime, _pool_buffers(destination_pool, config["page_bytes"])
        )
        if len(destination_buffers) != config["layers"]:
            raise RuntimeError("destination KV buffer registration count is incorrect")
        cache = CudaIpcMappingCache(runtime)
        copy_stream = torch.cuda.Stream(device=destination_pool[0].device)
        if config["submission_mode"] == "worker":
            transfer_worker = _BoundedTransferWorker(config["worker_queue_depth"])
        result_queue.put({"kind": "destination_ready"})

        compute_only = None
        if compute_command_queue is not None and compute_result_queue is not None:
            compute_only_trials = [
                _run_compute_only_trial(
                    compute_command_queue,
                    compute_result_queue,
                    f"compute-only-{index}",
                    config["compute_iterations"],
                )
                for index in range(config["compute_baseline_repetitions"])
            ]
            compute_only = {
                "trials": compute_only_trials,
                "cuda_median_ms": statistics.median(
                    trial["cuda_ms"] for trial in compute_only_trials
                ),
                "host_median_ns": statistics.median(
                    trial["done_host_ns"] - trial["start_host_ns"]
                    for trial in compute_only_trials
                ),
            }

        for generation_index in range(2):
            source = metadata_queue.get(timeout=config["timeout_s"])
            if source.get("kind") != "source_registration":
                raise RuntimeError(f"unexpected source message: {source}")

            source["pool_pages"] = config["pool_pages"]
            source["fragment_run_pages"] = config["fragment_run_pages"]
            cold_map_start = _mono_ns()
            mapped_bases = []
            cold_reused = []
            for allocation in source["allocations"]:
                mapped, reused = cache.map(
                    peer_instance_id=source["peer_instance_id"],
                    peer_session_id=source["peer_session_id"],
                    pool_generation=source["generation"],
                    ipc_handle=allocation.ipc_handle,
                )
                mapped_bases.append(mapped)
                cold_reused.append(reused)
            cold_mapping_us = (_mono_ns() - cold_map_start) / 1_000
            if any(cold_reused):
                raise RuntimeError(
                    "first mapping for a pool generation unexpectedly reused a cache entry"
                )

            rows = []
            for pattern in ("contiguous", "fragmented"):
                for requested_mib in config["sizes_mib"]:
                    requested_bytes = requested_mib * MIB
                    pages = math.ceil(
                        requested_bytes / (config["layers"] * config["page_bytes"])
                    )
                    source_indices, destination_indices = _page_indices(
                        config["pool_pages"],
                        pages,
                        pattern == "fragmented",
                        config["fragment_run_pages"],
                    )
                    for repeat in range(config["repetitions"]):
                        warm_map_start = _mono_ns()
                        warm_bases = []
                        warm_reused = []
                        for allocation in source["allocations"]:
                            mapped, reused = cache.map(
                                peer_instance_id=source["peer_instance_id"],
                                peer_session_id=source["peer_session_id"],
                                pool_generation=source["generation"],
                                ipc_handle=allocation.ipc_handle,
                            )
                            warm_bases.append(mapped)
                            warm_reused.append(reused)
                        warm_mapping_us = (_mono_ns() - warm_map_start) / 1_000
                        if not all(warm_reused) or warm_bases != mapped_bases:
                            raise RuntimeError("warm CUDA IPC mapping was not reused")
                        copy_args = dict(
                            runtime=runtime,
                            copy_stream=copy_stream,
                            destination_pool=destination_pool,
                            source_bases=warm_bases,
                            source_buffers=source["buffers"],
                            destination_buffers=destination_buffers,
                            source_indices=source_indices,
                            destination_indices=destination_indices,
                            generation_index=generation_index,
                        )
                        compute_started = None
                        compute_trial_id = (
                            f"overlap-{generation_index}-{pattern}-"
                            f"{requested_mib}-{repeat}"
                        )
                        destination_index_tensor = torch.as_tensor(
                            destination_indices,
                            dtype=torch.int64,
                            device=destination_pool[0].device,
                        )
                        for buffer in destination_pool:
                            buffer.index_fill_(0, destination_index_tensor, 0)
                        torch.cuda.synchronize(destination_pool[0].device)
                        if (
                            compute_command_queue is not None
                            and compute_result_queue is not None
                        ):
                            compute_started = _start_compute_trial(
                                compute_command_queue,
                                compute_result_queue,
                                compute_trial_id,
                                config["compute_iterations"],
                            )
                        caller_start_ns = _mono_ns()
                        if transfer_worker is None:
                            row = _copy_once(**copy_args)
                            worker_wait_ns = 0
                            caller_blocking = True
                            submit_return_ns = row["copy_complete_event_sync_host_ns"][
                                "done"
                            ]
                        else:
                            future, worker_wait_ns = transfer_worker.submit(
                                _copy_once, **copy_args
                            )
                            submit_return_ns = _mono_ns()
                            caller_blocking = False
                            row, scheduler_ticks = _wait_worker_future(
                                future, config["poll_sleep_us"]
                            )
                            row["scheduler_ticks_while_future_in_flight"] = (
                                scheduler_ticks
                            )
                        row["submission_mode"] = config["submission_mode"]
                        row["caller_blocking"] = caller_blocking
                        row["worker_wait_us"] = worker_wait_ns / 1_000
                        row["scheduler_submit_return_ns"] = submit_return_ns
                        row["caller_elapsed_us"] = (
                            _mono_ns() - caller_start_ns
                        ) / 1_000
                        if compute_started is not None:
                            assert compute_result_queue is not None
                            compute_done = _finish_compute_trial(
                                compute_result_queue, compute_trial_id
                            )
                            compute_trial = {**compute_started, **compute_done}
                            overlap_host_ns = (
                                compute_trial["done_host_ns"]
                                - compute_trial["start_host_ns"]
                            )
                            copy_start_ns = row["copy_start_event_record_host_ns"][
                                "start"
                            ]
                            copy_done_ns = row["copy_complete_event_sync_host_ns"][
                                "done"
                            ]
                            row["compute_overlap"] = {
                                "trial_id": compute_trial["trial_id"],
                                "cuda_ms": compute_trial["cuda_ms"],
                                "host_ns": overlap_host_ns,
                                "cuda_slowdown_ratio": compute_trial["cuda_ms"]
                                / max(1e-9, compute_only["cuda_median_ms"]),
                                "host_slowdown_ratio": overlap_host_ns
                                / max(1, compute_only["host_median_ns"]),
                                "host_intervals_overlap": max(
                                    copy_start_ns, compute_trial["start_host_ns"]
                                )
                                < min(copy_done_ns, compute_trial["done_host_ns"]),
                                "start_event_record_host_ns": compute_trial[
                                    "start_event_record_host_ns"
                                ],
                                "done_event_record_host_ns": compute_trial[
                                    "done_event_record_host_ns"
                                ],
                                "done_event_sync_host_ns": compute_trial[
                                    "done_event_sync_host_ns"
                                ],
                            }
                        row.update(
                            generation=source["generation"],
                            generation_index=generation_index,
                            pattern=pattern,
                            requested_mib=requested_mib,
                            actual_mib=row["bytes"] / MIB,
                            repeat=repeat,
                            warm_mapping_us=warm_mapping_us,
                        )
                        rows.append(row)

            failure_gate = _post_enqueue_failure_gate(
                runtime=runtime,
                destination_pool=destination_pool,
                destination_buffers=destination_buffers,
                source=source,
                mapped_bases=mapped_bases,
            )
            source_lifetime_gate = _source_reuse_lifetime_gate(
                runtime=runtime,
                destination_pool=destination_pool,
                destination_buffers=destination_buffers,
                source=source,
                mapped_bases=mapped_bases,
                command_queue=command_queue,
                metadata_queue=metadata_queue,
                delay_ms=config["lifetime_gate_delay_ms"],
            )
            if not failure_gate["gate_passed"]:
                raise RuntimeError("post-enqueue destination reuse gate failed")
            if not source_lifetime_gate["gate_passed"]:
                raise RuntimeError(
                    "source read-lease lifetime gate failed: "
                    f"{source_lifetime_gate}"
                )

            result_queue.put(
                {
                    "kind": "generation_complete",
                    "generation": source["generation"],
                    "source_registration_us": source["registration_us"],
                    "source_allocation_count": source["allocation_count"],
                    "source_buffer_count": source["buffer_count"],
                    "cold_mapping_us": cold_mapping_us,
                    "cold_mapping_reused": any(cold_reused),
                    "generation_changed_invalidated_previous_mapping": generation_index
                    > 0,
                    "source_base_addresses": source["base_addresses"],
                    "rows": rows,
                    "compute_only": compute_only,
                    "post_enqueue_failure_gate": failure_gate,
                    "source_reuse_lifetime_gate": source_lifetime_gate,
                }
            )
            command_queue.put({"action": "release", "generation": source["generation"]})
        cleanup_start = _mono_ns()
        if transfer_worker is not None:
            transfer_worker.close()
            transfer_worker = None
        cache.close()
        cache = None
        result_queue.put(
            {
                "kind": "destination_cleanup",
                "mapping_cleanup_us": (_mono_ns() - cleanup_start) / 1_000,
            }
        )
    except Exception as exc:  # noqa: BLE001 - report child-process failure
        result_queue.put({"kind": "error", "error": repr(exc)})
        command_queue.put({"action": "abort", "generation": "*"})
    finally:
        if transfer_worker is not None:
            transfer_worker.close()
        if cache is not None:
            cache.close()


def _compute_worker(
    gpu: int,
    matrix_size: int,
    command_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """Fixed-work compute-only and overlap trials on a distinct GPU."""
    try:
        torch.cuda.set_device(gpu)
        left = torch.randn(
            (matrix_size, matrix_size), dtype=torch.float16, device=f"cuda:{gpu}"
        )
        right = torch.randn_like(left)
        output = torch.empty_like(left)
        torch.mm(left, right, out=output)
        torch.cuda.synchronize(gpu)
        while True:
            command = command_queue.get()
            if command["action"] == "shutdown":
                return
            if command["action"] != "run":
                continue
            start = torch.cuda.Event(enable_timing=True)
            done = torch.cuda.Event(enable_timing=True)
            start_record_start = _mono_ns()
            start.record(torch.cuda.current_stream(gpu))
            start_record_done = _mono_ns()
            result_queue.put(
                {
                    "kind": "compute_started",
                    "trial_id": command["trial_id"],
                    "start_event_record_host_ns": {
                        "start": start_record_start,
                        "done": start_record_done,
                    },
                    "start_host_ns": start_record_done,
                }
            )
            for _ in range(command["iterations"]):
                torch.mm(left, right, out=output)
            done_record_start = _mono_ns()
            done.record(torch.cuda.current_stream(gpu))
            done_record_done = _mono_ns()
            sync_start = _mono_ns()
            done.synchronize()
            sync_done = _mono_ns()
            result_queue.put(
                {
                    "kind": "compute_done",
                    "trial_id": command["trial_id"],
                    "gpu": gpu,
                    "iterations": command["iterations"],
                    "matrix_size": matrix_size,
                    "cuda_ms": start.elapsed_time(done),
                    "start_host_ns": start_record_done,
                    "done_host_ns": sync_done,
                    "done_event_record_host_ns": {
                        "start": done_record_start,
                        "done": done_record_done,
                    },
                    "done_event_sync_host_ns": {
                        "start": sync_start,
                        "done": sync_done,
                    },
                }
            )
    except Exception as exc:  # noqa: BLE001 - report optional child failure
        result_queue.put({"kind": "compute_error", "error": repr(exc)})


def _join_owned(process: mp.Process, timeout_s: int, label: str) -> None:
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError(
            f"own {label} process exceeded {timeout_s}s and was terminated"
        )
    if process.exitcode:
        raise RuntimeError(f"{label} process exited with code {process.exitcode}")


def _get_result(result_queue: mp.Queue, timeout_s: int) -> dict[str, Any]:
    try:
        result = result_queue.get(timeout=timeout_s)
    except queue.Empty as exc:
        raise RuntimeError("timed out waiting for benchmark result") from exc
    if result["kind"] == "error":
        raise RuntimeError(result["error"])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gpu", type=int, default=0)
    parser.add_argument("--destination-gpu", type=int, default=1)
    parser.add_argument(
        "--sizes-mib",
        type=_parse_mib_list,
        default=[4, 256, 1757],
        help="Comma-separated representative transfer sizes; include long-context-equivalent sizes here.",
    )
    parser.add_argument("--layers", type=int, default=72)
    parser.add_argument("--page-bytes", type=int, default=2048)
    parser.add_argument(
        "--pool-pages",
        type=int,
        default=0,
        help="0 derives enough pages for fragmented largest size",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--poll-sleep-us", type=int, default=0)
    parser.add_argument("--fragment-run-pages", type=int, default=40)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument(
        "--submission-mode",
        choices=("synchronous", "worker"),
        default="worker",
        help="caller-blocking or bounded transfer-worker submission",
    )
    parser.add_argument("--worker-queue-depth", type=int, default=1)
    parser.add_argument("--compute-iterations", type=int, default=32)
    parser.add_argument("--compute-baseline-repetitions", type=int, default=3)
    parser.add_argument("--compute-matrix-size", type=int, default=4096)
    parser.add_argument("--lifetime-gate-delay-ms", type=int, default=100)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--overlap-gpu",
        type=int,
        default=-1,
        help="Optional third, distinct GPU for independent compute overlap",
    )
    return parser.parse_args()


def _environment(gpu_ordinals: list[int]) -> dict[str, Any]:
    runtime_version = None
    try:
        runtime_version = CudaIpcRuntime().runtime_version
    except Exception as exc:  # noqa: BLE001 - include unavailable runtime in raw JSON
        runtime_version = f"unavailable: {exc}"
    devices = []
    for ordinal in sorted(set(gpu_ordinals)):
        properties = torch.cuda.get_device_properties(ordinal)
        devices.append(
            {
                "ordinal": ordinal,
                "name": properties.name,
                "uuid": str(getattr(properties, "uuid", "")),
                "total_memory": properties.total_memory,
                "major": properties.major,
                "minor": properties.minor,
            }
        )
    return {
        "sys_executable": sys.executable,
        "python": sys.version,
        "torch": torch.__version__,
        "built_cuda": torch.version.cuda,
        "actual_runtime_version": runtime_version,
        "cuda_runtime_library": os.environ.get("DT_CUDA_RUNTIME_LIBRARY", ""),
        "gpu_devices": devices,
    }


def main() -> None:
    args = parse_args()
    if args.source_gpu == args.destination_gpu:
        raise SystemExit("source and destination GPUs must differ")
    if args.layers <= 0 or args.page_bytes <= 0 or args.repetitions <= 0:
        raise SystemExit("layers, page bytes, and repetitions must be positive")
    if args.overlap_gpu in (args.source_gpu, args.destination_gpu):
        raise SystemExit("--overlap-gpu must be distinct from source and destination")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; this benchmark has no CPU simulation")
    if (
        max(args.source_gpu, args.destination_gpu, args.overlap_gpu)
        >= torch.cuda.device_count()
    ):
        raise SystemExit("requested GPU ordinal is unavailable")
    if (
        args.worker_queue_depth <= 0
        or args.compute_iterations <= 0
        or args.compute_baseline_repetitions <= 0
        or args.compute_matrix_size <= 0
        or args.fragment_run_pages <= 0
        or args.lifetime_gate_delay_ms <= 0
    ):
        raise SystemExit(
            "worker queue depth, compute iterations/baseline repetitions, matrix "
            "size, and fragment run pages must be positive"
        )

    largest_pages = math.ceil(
        max(args.sizes_mib) * MIB / (args.layers * args.page_bytes)
    )
    pool_pages = args.pool_pages or largest_pages * 2
    if pool_pages < largest_pages * 2:
        raise SystemExit("--pool-pages must accommodate fragmented largest transfer")
    context = mp.get_context("spawn")
    peer_instance_id = f"dt-ipc-feasibility-{uuid.uuid4()}"
    config = {
        "source_gpu": args.source_gpu,
        "destination_gpu": args.destination_gpu,
        "layers": args.layers,
        "page_bytes": args.page_bytes,
        "pool_pages": pool_pages,
        "sizes_mib": args.sizes_mib,
        "repetitions": args.repetitions,
        "poll_sleep_us": args.poll_sleep_us,
        "timeout_s": args.timeout_s,
        "submission_mode": args.submission_mode,
        "worker_queue_depth": args.worker_queue_depth,
        "compute_iterations": args.compute_iterations,
        "compute_baseline_repetitions": args.compute_baseline_repetitions,
        "compute_matrix_size": args.compute_matrix_size,
        "fragment_run_pages": args.fragment_run_pages,
        "lifetime_gate_delay_ms": args.lifetime_gate_delay_ms,
        "pattern_chunk_pages": 2048,
        "peer_instance_id": peer_instance_id,
    }

    metadata_queue = context.Queue()
    command_queue = context.Queue()
    result_queue = context.Queue()
    compute_command_queue = context.Queue() if args.overlap_gpu >= 0 else None
    compute_result_queue = context.Queue() if args.overlap_gpu >= 0 else None
    compute_process = None
    if args.overlap_gpu >= 0:
        compute_process = context.Process(
            target=_compute_worker,
            args=(
                args.overlap_gpu,
                args.compute_matrix_size,
                compute_command_queue,
                compute_result_queue,
            ),
            name="dt-ipc-overlap-compute",
        )
        compute_process.start()

    destination = context.Process(
        target=_destination_worker,
        args=(
            config,
            metadata_queue,
            command_queue,
            result_queue,
            compute_command_queue,
            compute_result_queue,
        ),
        name="dt-ipc-destination",
    )
    destination.start()
    source: mp.Process | None = None
    try:
        ready = _get_result(result_queue, args.timeout_s)
        if ready["kind"] != "destination_ready":
            raise RuntimeError(f"unexpected destination startup result: {ready}")
        generations = ["generation-1", "generation-2-restart"]
        all_results = []
        for generation in generations:
            source = context.Process(
                target=_source_worker,
                args=(
                    config,
                    generation,
                    len(all_results),
                    metadata_queue,
                    command_queue,
                ),
                name=f"dt-ipc-source-{generation}",
            )
            source.start()
            result = _get_result(result_queue, args.timeout_s)
            if result["kind"] != "generation_complete":
                raise RuntimeError(f"unexpected destination result: {result}")
            _join_owned(source, args.timeout_s, generation)
            source = None
            all_results.append(result)
        _join_owned(destination, args.timeout_s, "destination")
        cleanup = _get_result(result_queue, args.timeout_s)
        if cleanup["kind"] != "destination_cleanup":
            raise RuntimeError(f"unexpected destination cleanup result: {cleanup}")

        if compute_process is not None:
            compute_command_queue.put({"action": "shutdown"})
            _join_owned(compute_process, args.timeout_s, "overlap compute")
        raw = {
            "backend": "cuda_ipc_p2p_batch",
            "command": [sys.executable, *sys.argv],
            "parameters": {
                "sizes_mib": args.sizes_mib,
                "repetitions": args.repetitions,
                "poll_sleep_us": args.poll_sleep_us,
                "timeout_s": args.timeout_s,
                "worker_queue_depth": args.worker_queue_depth,
                "compute_iterations": args.compute_iterations,
                "compute_baseline_repetitions": args.compute_baseline_repetitions,
                "compute_matrix_size": args.compute_matrix_size,
                "fragment_run_pages": args.fragment_run_pages,
                "lifetime_gate_delay_ms": args.lifetime_gate_delay_ms,
            },
            "source_gpu": args.source_gpu,
            "destination_gpu": args.destination_gpu,
            "layers": args.layers,
            "page_bytes": args.page_bytes,
            "pool_pages": pool_pages,
            "submission_mode": args.submission_mode,
            "environment": _environment(
                [args.source_gpu, args.destination_gpu]
                + ([args.overlap_gpu] if args.overlap_gpu >= 0 else [])
            ),
            "process_restart_invalidation": {
                "first_generation": all_results[0]["generation"],
                "second_generation": all_results[1]["generation"],
                "second_cold_mapping_reused": all_results[1]["cold_mapping_reused"],
                "first_source_bases": all_results[0]["source_base_addresses"],
                "second_source_bases": all_results[1]["source_base_addresses"],
            },
            "cleanup": cleanup,
            "generations": all_results,
        }
        rendered = json.dumps(raw, indent=2, sort_keys=True)
        print(rendered)
        if args.output_json is not None:
            args.output_json.write_text(rendered + "\n")
    finally:
        if source is not None and source.is_alive():
            command_queue.put({"action": "abort", "generation": "*"})
            _join_owned(source, args.timeout_s, "source")
        if destination.is_alive():
            destination.terminate()
            destination.join()
        if compute_process is not None and compute_process.is_alive():
            compute_command_queue.put({"action": "shutdown"})
            _join_owned(compute_process, args.timeout_s, "overlap compute")


if __name__ == "__main__":
    # Spawn is required: CUDA IPC cannot safely inherit an initialized CUDA runtime.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
