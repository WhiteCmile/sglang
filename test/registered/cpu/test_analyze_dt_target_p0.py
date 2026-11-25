import importlib.util
import json
from pathlib import Path

import sglang.srt


def _module():
    path = Path(__file__).parents[3] / "scripts" / "analyze_dt_target_p0.py"
    spec = importlib.util.spec_from_file_location("analyze_dt_target_p0", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_p0_join_and_recommendation(tmp_path):
    module = _module()
    path = tmp_path / "target.jsonl"
    events = [
        dict(
            event="target_p0_hop",
            hop="feature_published",
            transfer_id="t1",
            hidden_produced_ns=100,
            descriptor_published_ns=150,
        ),
        dict(
            event="target_p0_hop",
            hop="record_visible",
            transfer_id="t1",
            hidden_produced_ns=9_999_999,
            hidden_produced_host_ns=100,
            record_visible_ns=125,
        ),
        dict(
            event="target_p0_hop",
            hop="dispatch_registered",
            transfer_id="t1",
            ledger_begin_finished_ns=300,
        ),
        dict(
            event="target_p0_hop",
            hop="candidate_observed",
            transfer_id="t1",
            completion_observed_ns=1000,
        ),
        dict(
            event="target_p0_hop",
            hop="ready_snapshot",
            transfer_id="t1",
            completion_observed_ns=1000,
            ready_snapshot_ns=1020,
            tp_broadcast_end_ns=1040,
        ),
        dict(
            event="target_p0_hop",
            hop="tp_ready_applied",
            transfer_id="t1",
            completion_observed_ns=1000,
            tp_apply_ns=1050,
        ),
        dict(
            event="target_p0_hop",
            hop="frozen_for_verify",
            transfer_id="t1",
            freeze_ns=1100,
            candidate_slots_total=8,
            candidate_slots_free=5,
            source_pages_total=512,
            source_pages_free=500,
            ring_credit_free=4,
        ),
        dict(
            event="target_p0_hop",
            hop="frozen_for_verify",
            transfer_id="t2",
            freeze_ns=1075,
        ),
        dict(
            event="target_p0_hop",
            hop="verify_submitted",
            transfer_id="t1",
            submission_returned_ns=1200,
            control_phase_ns=20,
            poll_phase_ns=30,
        ),
        dict(
            event="target_p0_hop",
            hop="verify_gpu_end",
            transfer_id="t1",
            gpu_end_observed_ns=1500,
        ),
        dict(event="target_p0_hop", hop="at_launch_fallback", reason="credit"),
    ]
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    loaded, fallback = module.load_events([path])
    rows = module.build_rows(loaded)
    summary = module.summarize(rows, fallback)
    assert rows[0]["p1_p2_corridor_ns"] == 200
    assert rows[0]["hidden_to_record_visible_ns"] == 25
    assert rows[0]["p3_corridor_ns"] == 100
    assert rows[0]["candidate_to_ready_snapshot_ns"] == 20
    assert rows[0]["ready_snapshot_to_tp_end_ns"] == 20
    assert rows[0]["missed_freeze_windows"] == 1
    assert summary["recommendation"] == "P1+P2"
    assert summary["counters"]["at_launch_credit_fallbacks"] == 1
    assert summary["counters"]["candidate_slots_high_water"] == 3
    assert summary["counters"]["source_pages_high_water"] == 12
    assert summary["counters"]["rows_missing_at_least_one_freeze"] == 1


def test_flashmtp_runtime_wires_the_send_side_p0_boundary():
    """Prevent a DFlash-only P0 trace from passing on FlashMTP CI again."""

    root = Path(next(iter(sglang.srt.__path__)))
    target = (root / "speculative" / "dt_target.py").read_text(encoding="utf-8")
    worker = (root / "speculative" / "flashmtp_worker_v2.py").read_text(
        encoding="utf-8"
    )
    target_worker = (
        root / "speculative" / "flashmtp_target_worker_v2.py"
    ).read_text(encoding="utf-8")

    pivot_start = target.index("    def _stage_flashmtp_pivot_copy(")
    pivot_end = target.index("\n    def ", pivot_start + 1)
    pivot = target[pivot_start:pivot_end]
    assert 'self._p0_emit(\n                                "feature_published"' in pivot
    assert 'source_kind="flashmtp_pivot"' in pivot

    # Decode and prefill must both stamp the producer boundary for FlashMTP,
    # not only for DFlash.
    assert worker.count("batch_result.dt_hidden_produced_ns = target_done_ns") == 1
    assert 'if not getattr(self, "_is_token_only", False):' in worker
    assert "if not self._is_token_only:" in target_worker
    assert "batch_output.dt_hidden_produced_ns = time.monotonic_ns()" in target_worker
