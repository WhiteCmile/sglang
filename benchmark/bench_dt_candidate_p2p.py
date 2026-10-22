"""Bounded two-process gate for direct DT candidate notification and reuse.

Run this only after the project GPU ownership check. Candidate rows remain on
GPU throughout: a Draft process writes Target-owned CUDA IPC rows and reports
its stream completion through the host transport, after which the Target
performs only a tiny metadata D2H and event query. The gate covers early data
completion, independent entries, fixed graph input reuse, generation reuse,
and notification under unrelated compute.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT / "sglang" / "python", _ROOT / "dt-runtime" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import msgspec  # noqa: E402
import torch  # noqa: E402
from dt_protocol import (  # noqa: E402
    CandidateDestinationRef,
    DraftAssignment,
)
from sglang.srt.speculative.dt_candidate_p2p import (  # noqa: E402
    CandidateTransportError,
    DraftCandidatePushTransport,
    TargetCandidateArena,
)


def _mono_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def _task(raw: dict, destination: CandidateDestinationRef):
    return SimpleNamespace(
        transfer_id=raw["transfer_id"],
        target_instance_id=raw["target_instance_id"],
        next_token_id=raw["root_token"],
        candidate_destination=destination,
    )


def _draft_worker(config: dict, commands: mp.Queue, results: mp.Queue) -> None:
    try:
        torch.cuda.set_device(config["draft_gpu"])
        transport = DraftCandidatePushTransport(
            draft_instance_id=config["draft_instance_id"],
            draft_session_id=config["draft_session_id"],
            device=torch.device("cuda", config["draft_gpu"]),
            profile_timing=config["profile_timing"],
        )
        registration = commands.get()
        if registration.get("action") != "register":
            raise ValueError("Draft prototype requires arena registration first")
        from dt_protocol import CandidateArenaRegistration

        transport.register(
            CandidateArenaRegistration(**registration["registration"])
        )
        results.put({"kind": "ready", "pid": os.getpid()})
        while True:
            command = commands.get()
            if command["action"] == "stop":
                transport.close()
                results.put({"kind": "stopped"})
                return
            if command["action"] not in {"push_wave", "cancel_wave"}:
                raise ValueError(f"unknown Draft prototype action {command['action']!r}")
            for row in command["rows"]:
                delay_s = float(row.get("delay_s", 0.0))
                if delay_s:
                    time.sleep(delay_s)
                destination = CandidateDestinationRef(**row["destination"])
                task = _task(row, destination)
                with torch.cuda.stream(transport.push_stream):
                    if command["action"] == "cancel_wave":
                        push = transport.publish_cancelled([task])
                    else:
                        candidates = torch.full(
                            (1, destination.row_width),
                            int(row["value"]),
                            dtype=torch.int32,
                            device=f"cuda:{config['draft_gpu']}",
                        )
                        candidates[0, 0] = task.next_token_id
                        push = transport.submit(tasks=[task], candidates=candidates)
                results.put(
                    {
                        "kind": "submitted",
                        "transfer_id": task.transfer_id,
                        "submitted_ns": push.submitted_ns,
                        "cold_mapping_count": push.cold_mapping_count,
                    }
                )
                push.event.synchronize()
                push_cuda_ns = (
                    int(push.start_event.elapsed_time(push.event) * 1_000_000)
                    if push.start_event is not None
                    else None
                )
                transport.publish_completion(
                    [task], cancelled=command["action"] == "cancel_wave"
                )
                completion_deadline = time.monotonic() + config["timeout_s"]
                while transport.pending_completion_count:
                    transport.progress_completions()
                    if time.monotonic() >= completion_deadline:
                        raise TimeoutError(
                            f"direct completion ACK timed out: {task.transfer_id}"
                        )
                    time.sleep(0)
                results.put(
                    {
                        "kind": (
                            "cancel_acked"
                            if command["action"] == "cancel_wave"
                            else "acked"
                        ),
                        "transfer_id": task.transfer_id,
                        "draft_instance_id": config["draft_instance_id"],
                        "push_cuda_ns": push_cuda_ns,
                        "draft_session_id": config["draft_session_id"],
                        "completed_ns": _mono_ns(),
                    }
                )
    except Exception as exc:  # noqa: BLE001 - child evidence must retain failure
        results.put(
            {
                "kind": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _recv(results: mp.Queue, *, timeout_s: float) -> dict:
    try:
        message = results.get(timeout=timeout_s)
    except queue.Empty as exc:
        raise TimeoutError("candidate prototype child timed out") from exc
    if message.get("kind") == "error":
        raise RuntimeError(message["error"] + "\n" + message["traceback"])
    return message


def _pump_host_completions(
    arena: TargetCandidateArena, results: mp.Queue
) -> None:
    arena.poll_completion_notices(max_messages=arena.slot_count * 2)
    while True:
        try:
            message = results.get_nowait()
        except queue.Empty:
            return
        if message.get("kind") == "error":
            raise RuntimeError(message["error"] + "\n" + message["traceback"])


def _wait_completion(
    arena: TargetCandidateArena,
    results: mp.Queue,
    transfer_id: str,
    *,
    timeout_s: float,
):
    started_ns = _mono_ns()
    queries = 0
    while _mono_ns() - started_ns < int(timeout_s * 1_000_000_000):
        queries += 1
        _pump_host_completions(arena, results)
        for completion in arena.poll(max_queries=arena.slot_count):
            if completion.transfer_id == transfer_id:
                return completion, queries, _mono_ns() - started_ns
        time.sleep(0)
    raise TimeoutError(f"candidate completion timed out: {transfer_id}")


def _release(arena: TargetCandidateArena, completion, *, timeout_s: float) -> None:
    started_ns = _mono_ns()
    while _mono_ns() - started_ns < int(timeout_s * 1_000_000_000):
        if arena.poll_releases():
            return
        time.sleep(0)
    raise TimeoutError("candidate read fence did not release its slot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-gpu", type=int, default=6)
    parser.add_argument("--draft-gpu", type=int, default=7)
    parser.add_argument("--row-width", type=int, default=8)
    parser.add_argument("--reuse-rounds", type=int, default=8)
    parser.add_argument("--slow-entry-ms", type=float, default=40.0)
    parser.add_argument("--compute-size", type=int, default=1024)
    parser.add_argument("--compute-replays", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--profile-timing", action="store_true")
    args = parser.parse_args()
    if args.target_gpu == args.draft_gpu:
        parser.error("the visibility gate requires distinct Target and Draft GPUs")
    if args.row_width <= 1 or args.reuse_rounds <= 0 or args.compute_size <= 0:
        parser.error("prototype dimensions must be positive")

    config = {
        "draft_gpu": args.draft_gpu,
        "draft_instance_id": "candidate-prototype-draft",
        "draft_session_id": "candidate-prototype-draft-session",
        "timeout_s": args.timeout_s,
        "profile_timing": args.profile_timing,
    }
    context = mp.get_context("spawn")
    commands = context.Queue()
    results = context.Queue()
    child = context.Process(
        target=_draft_worker, args=(config, commands, results), daemon=False
    )
    child.start()

    torch.cuda.set_device(args.target_gpu)
    device = torch.device("cuda", args.target_gpu)
    arena = TargetCandidateArena(
        target_instance_id="candidate-prototype-target",
        target_session_id="candidate-prototype-target-session",
        device=device,
        slot_count=2,
        row_width=args.row_width,
        profile_timing=args.profile_timing,
    )
    fixed_graph_input = torch.empty((1, args.row_width), dtype=torch.int32, device=device)
    graph_output = torch.empty_like(fixed_graph_input)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.add(fixed_graph_input, 1, out=graph_output)
    fixed_address = fixed_graph_input.data_ptr()
    evidence = {
        "schema_version": "dt-candidate-p2p-prototype/v3-direct-host-completion",
        "target_gpu": args.target_gpu,
        "draft_gpu": args.draft_gpu,
        "target_pid": os.getpid(),
        "draft_pid": None,
        "row_width": args.row_width,
        "slot_count": arena.slot_count,
        "fixed_graph_input_address": fixed_address,
        "waves": [],
    }

    try:
        commands.put(
            {
                "action": "register",
                "registration": msgspec.to_builtins(
                    arena.registration(), builtin_types=(bytes,)
                ),
            }
        )
        ready = _recv(results, timeout_s=args.timeout_s)
        if ready.get("kind") != "ready":
            raise RuntimeError(f"unexpected Draft startup message: {ready}")
        evidence["draft_pid"] = ready["pid"]

        # Send the fast entry first but label the second entry slow.  Both waits
        # are armed together; observing fast while slow remains pending proves
        # that notification streams do not introduce a shared FIFO barrier.
        fast = arena.reserve("wave-fast")
        slow = arena.reserve("wave-slow")
        for destination in (fast, slow):
            arena.authorize(
                DraftAssignment(
                    transfer_id=destination.transfer_id,
                    draft_instance_id=config["draft_instance_id"],
                    draft_session_id="candidate-prototype-draft-session",
                    authorization_nonce=destination.authorization_nonce,
                )
            )
        commands.put(
            {
                "action": "push_wave",
                "rows": [
                    {
                        "transfer_id": fast.transfer_id,
                        "target_instance_id": fast.target_instance_id,
                        "root_token": 101,
                        "value": 11,
                        "destination": msgspec.to_builtins(
                            fast, builtin_types=(bytes,)
                        ),
                    },
                    {
                        "transfer_id": slow.transfer_id,
                        "target_instance_id": slow.target_instance_id,
                        "root_token": 202,
                        "value": 22,
                        "delay_s": args.slow_entry_ms / 1_000,
                        "destination": msgspec.to_builtins(
                            slow, builtin_types=(bytes,)
                        ),
                    },
                ],
            }
        )
        fast_completion, fast_queries, fast_wait_ns = _wait_completion(
            arena, results, fast.transfer_id, timeout_s=args.timeout_s
        )
        if any(item.transfer_id == slow.transfer_id for item in arena.poll()):
            raise RuntimeError("slow entry unexpectedly completed with the fast entry")
        fast_rows = arena.begin_read([fast_completion])
        fixed_graph_input.copy_(fast_rows)
        graph.replay()
        arena.finish_read([fast_completion])
        expected = torch.full_like(graph_output, 12)
        expected[0, 0] = 102
        if not torch.equal(graph_output, expected):
            raise RuntimeError("fast candidate was not visible to fixed graph input")
        _release(arena, fast_completion, timeout_s=args.timeout_s)

        slow_completion, slow_queries, slow_wait_ns = _wait_completion(
            arena, results, slow.transfer_id, timeout_s=args.timeout_s
        )
        slow_rows = arena.begin_read([slow_completion])
        fixed_graph_input.copy_(slow_rows)
        graph.replay()
        arena.finish_read([slow_completion])
        expected.fill_(23)
        expected[0, 0] = 203
        if not torch.equal(graph_output, expected):
            raise RuntimeError("slow candidate was not visible to fixed graph input")
        _release(arena, slow_completion, timeout_s=args.timeout_s)
        evidence["waves"].append(
            {
                "kind": "independent_host_completions",
                "fast_queries": fast_queries,
                "fast_wait_ns": fast_wait_ns,
                "slow_queries": slow_queries,
                "slow_wait_ns": slow_wait_ns,
                "slow_delay_ms": args.slow_entry_ms,
            }
        )

        previous_generation = 0
        stale_completion = slow_completion
        compute_stream = torch.cuda.Stream(device=device)
        compute_left = torch.randn(
            (args.compute_size, args.compute_size), dtype=torch.float16, device=device
        )
        compute_right = torch.randn_like(compute_left)
        for round_index in range(args.reuse_rounds):
            transfer_id = f"reuse-{round_index}"
            destination = arena.reserve(transfer_id)
            if destination.generation <= previous_generation:
                raise RuntimeError("candidate slot generation did not advance")
            previous_generation = destination.generation
            if round_index == 0:
                # Data completion intentionally precedes assignment metadata.
                authorize_late = True
            else:
                authorize_late = False
                arena.authorize(
                    DraftAssignment(
                        transfer_id=transfer_id,
                        draft_instance_id=config["draft_instance_id"],
                        draft_session_id="candidate-prototype-draft-session",
                        authorization_nonce=destination.authorization_nonce,
                    )
                )
            with torch.cuda.stream(compute_stream):
                for _ in range(args.compute_replays):
                    torch.mm(compute_left, compute_right)
            root_token = 300 + round_index
            value = 40 + round_index
            commands.put(
                {
                    "action": "push_wave",
                    "rows": [
                        {
                            "transfer_id": transfer_id,
                            "target_instance_id": destination.target_instance_id,
                            "root_token": root_token,
                            "value": value,
                            "destination": msgspec.to_builtins(
                                destination, builtin_types=(bytes,)
                            ),
                        }
                    ],
                }
            )
            if authorize_late:
                deadline_ns = _mono_ns() + int(args.timeout_s * 1_000_000_000)
                record_words = getattr(arena, "_completion_words", None)
                if record_words is None:
                    while arena._slots[destination.slot_id].state.value == "armed":
                        _pump_host_completions(arena, results)
                        arena.poll()
                        if _mono_ns() >= deadline_ns:
                            raise TimeoutError(
                                "early candidate completion was not observed"
                            )
                        time.sleep(0)
                else:
                    # Completion records authenticate against the assignment:
                    # the expectation is armed by authorize(), so a record
                    # that beats it is consumed on the first later poll
                    # instead of parking the slot.  Wait for the producer's
                    # 64-bit nonce release, then prove nothing is exposed
                    # before the assignment.
                    nonce = destination.authorization_nonce
                    while int(record_words[destination.slot_id, 0]) != nonce:
                        _pump_host_completions(arena, results)
                        arena.poll()
                        if _mono_ns() >= deadline_ns:
                            raise TimeoutError(
                                "early completion record was not observed"
                            )
                        time.sleep(0)
                if arena.poll():
                    raise RuntimeError("unassigned candidate completion was exposed")
                arena.authorize(
                    DraftAssignment(
                        transfer_id=transfer_id,
                        draft_instance_id=config["draft_instance_id"],
                        draft_session_id="candidate-prototype-draft-session",
                        authorization_nonce=destination.authorization_nonce,
                    )
                )
            completion, queries, wait_ns = _wait_completion(
                arena, results, transfer_id, timeout_s=args.timeout_s
            )
            rows = arena.begin_read([completion])
            if fixed_graph_input.data_ptr() != fixed_address:
                raise RuntimeError("fixed graph input address changed")
            fixed_graph_input.copy_(rows)
            graph.replay()
            arena.finish_read([completion])
            expected.fill_(value + 1)
            expected[0, 0] = root_token + 1
            if not torch.equal(graph_output, expected):
                raise RuntimeError("reused candidate row failed visibility validation")
            _release(arena, completion, timeout_s=args.timeout_s)
            evidence["waves"].append(
                {
                    "kind": "reuse",
                    "round": round_index,
                    "slot_id": destination.slot_id,
                    "generation": destination.generation,
                    "assignment_after_completion": authorize_late,
                    "queries": queries,
                    "host_completion_observe_ns": wait_ns,
                }
            )

        stale_rejected = False
        try:
            arena.begin_read([stale_completion])
        except CandidateTransportError:
            stale_rejected = True
        if not stale_rejected:
            raise RuntimeError("stale candidate completion was accepted after reuse")

        unsubmitted_cancel = arena.reserve("cancel-unsubmitted")
        if not arena.cancel(
            unsubmitted_cancel.transfer_id, remote_quiescent=True
        ):
            raise RuntimeError("unsubmitted cancellation did not release immediately")
        unassigned_reuse = arena.reserve("cancel-unsubmitted-reuse")
        if (
            unassigned_reuse.slot_id != unsubmitted_cancel.slot_id
            or unassigned_reuse.generation <= unsubmitted_cancel.generation
        ):
            raise RuntimeError("unsubmitted cancellation did not safely reuse its slot")
        arena.cancel(unassigned_reuse.transfer_id, remote_quiescent=True)

        preack_cancel = arena.reserve("cancel-before-assignment-ack")
        if arena.cancel(preack_cancel.transfer_id):
            raise RuntimeError("pre-ACK cancellation reused without remote proof")
        arena.authorize(
            DraftAssignment(
                transfer_id=preack_cancel.transfer_id,
                draft_instance_id=config["draft_instance_id"],
                draft_session_id=config["draft_session_id"],
                authorization_nonce=preack_cancel.authorization_nonce,
            )
        )
        commands.put(
            {
                "action": "cancel_wave",
                "rows": [
                    {
                        "transfer_id": preack_cancel.transfer_id,
                        "target_instance_id": preack_cancel.target_instance_id,
                        "root_token": 0,
                        "value": 0,
                        "destination": msgspec.to_builtins(
                            preack_cancel, builtin_types=(bytes,)
                        ),
                    }
                ],
            }
        )
        cancel_deadline = _mono_ns() + int(args.timeout_s * 1_000_000_000)
        while arena.active_count:
            _pump_host_completions(arena, results)
            arena.poll_cancelled()
            if _mono_ns() >= cancel_deadline:
                raise TimeoutError("pre-ACK cancellation tombstone was not observed")
            time.sleep(0)
        post_preack_reuse = arena.reserve("cancel-preack-post-reuse")
        if (
            post_preack_reuse.slot_id != preack_cancel.slot_id
            or post_preack_reuse.generation <= preack_cancel.generation
        ):
            raise RuntimeError("pre-ACK cancellation did not safely reuse its slot")
        arena.cancel(post_preack_reuse.transfer_id, remote_quiescent=True)

        assigned_cancel = arena.reserve("cancel-assigned")
        arena.authorize(
            DraftAssignment(
                transfer_id=assigned_cancel.transfer_id,
                draft_instance_id=config["draft_instance_id"],
                draft_session_id=config["draft_session_id"],
                authorization_nonce=assigned_cancel.authorization_nonce,
            )
        )
        if arena.cancel(assigned_cancel.transfer_id):
            raise RuntimeError("assigned cancellation reused before quiescence proof")
        commands.put(
            {
                "action": "cancel_wave",
                "rows": [
                    {
                        "transfer_id": assigned_cancel.transfer_id,
                        "target_instance_id": assigned_cancel.target_instance_id,
                        "root_token": 0,
                        "value": 0,
                        "destination": msgspec.to_builtins(
                            assigned_cancel, builtin_types=(bytes,)
                        ),
                    }
                ],
            }
        )
        cancel_deadline = _mono_ns() + int(args.timeout_s * 1_000_000_000)
        while arena.active_count:
            _pump_host_completions(arena, results)
            arena.poll_cancelled()
            if _mono_ns() >= cancel_deadline:
                raise TimeoutError("assigned cancellation tombstone was not observed")
            time.sleep(0)
        post_cancel_reuse = arena.reserve("cancel-post-reuse")
        if (
            post_cancel_reuse.slot_id != assigned_cancel.slot_id
            or post_cancel_reuse.generation <= assigned_cancel.generation
        ):
            raise RuntimeError("assigned cancellation did not safely reuse its slot")
        arena.cancel(post_cancel_reuse.transfer_id, remote_quiescent=True)
        evidence["cancellation"] = {
            "unsubmitted_reused": True,
            "pre_assignment_ack_quarantined_until_tombstone": True,
            "pre_assignment_ack_reused_after_tombstone": True,
            "assigned_quarantined_until_tombstone": True,
            "assigned_reused_after_tombstone": True,
            "post_cancel_generation": post_cancel_reuse.generation,
        }
        evidence["stale_completion_rejected"] = True
        evidence["fixed_graph_input_address_stable"] = (
            fixed_graph_input.data_ptr() == fixed_address
        )
        evidence["active_slots"] = arena.active_count
        evidence["quarantined_slots"] = arena.quarantined_slot_count
        evidence["passed"] = bool(
            evidence["fixed_graph_input_address_stable"]
            and evidence["stale_completion_rejected"]
            and evidence["active_slots"] == 0
            and evidence["quarantined_slots"] == 0
        )
        if not evidence["passed"]:
            raise RuntimeError("candidate prototype did not reach a clean pass state")
        print(json.dumps(evidence, indent=2, sort_keys=True))
    finally:
        commands.put({"action": "stop"})
        child.join(timeout=args.timeout_s)
        if child.is_alive():
            child.terminate()
            child.join(timeout=2)
        arena.close()


if __name__ == "__main__":
    main()
