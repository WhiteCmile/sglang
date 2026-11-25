from __future__ import annotations

import random

import pytest
from sglang.srt.speculative.dt_source_page_allocator import (
    SourceExtentLease,
    SourcePageAllocationError,
    TargetSourcePageAllocator,
)


def test_small_waves_share_one_physical_source_slot() -> None:
    allocator = TargetSourcePageAllocator(
        slot_count=4, slot_rows=8192, page_rows=64
    )

    leases = [allocator.allocate(f"wave-{index}", 48) for index in range(32)]

    assert {lease.slot_id for lease in leases} == {0}
    assert {lease.slot_generation for lease in leases} == {1}
    assert allocator.active_lease_count == 32
    assert allocator.free_page_count == 4 * 128 - 32
    assert allocator.internal_fragmentation_rows == 32 * (64 - 48)


def test_out_of_order_release_coalesces_pages() -> None:
    allocator = TargetSourcePageAllocator(
        slot_count=1, slot_rows=256, page_rows=32
    )
    leases = [allocator.allocate(str(index), 32) for index in range(8)]

    for index in (1, 3, 2):
        allocator.release(
            str(index),
            slot_id=leases[index].slot_id,
            slot_generation=leases[index].slot_generation,
        )

    assert allocator.largest_free_extent_pages == 3
    two_pages = allocator.allocate("two-pages", 64)
    assert two_pages.page_start == 1
    assert two_pages.page_count == 2


def test_slot_generation_advances_only_after_full_drain() -> None:
    allocator = TargetSourcePageAllocator(
        slot_count=1, slot_rows=128, page_rows=32
    )
    first = allocator.allocate("first", 32)
    allocator.allocate("second", 32)
    allocator.release("first")
    same_epoch = allocator.allocate("same-epoch", 32)

    assert same_epoch.slot_generation == first.slot_generation

    allocator.release("second")
    allocator.release("same-epoch")
    next_epoch = allocator.allocate("next-epoch", 32)
    assert next_epoch.slot_generation == first.slot_generation + 1


def test_stale_read_proof_cannot_release_reused_pages() -> None:
    allocator = TargetSourcePageAllocator(
        slot_count=1, slot_rows=64, page_rows=64
    )
    old = allocator.allocate("old", 1)
    allocator.release(
        "old", slot_id=old.slot_id, slot_generation=old.slot_generation
    )
    new = allocator.allocate("new", 1)

    with pytest.raises(SourcePageAllocationError, match="stale source slot generation"):
        allocator.release(
            "new", slot_id=old.slot_id, slot_generation=old.slot_generation
        )
    assert allocator.get("new") == new


def test_one_lease_never_crosses_physical_slot_boundary() -> None:
    allocator = TargetSourcePageAllocator(
        slot_count=2, slot_rows=128, page_rows=32
    )
    allocator.allocate("left-a", 64)
    allocator.allocate("left-b", 32)
    allocator.allocate("right-a", 96)

    with pytest.raises(SourcePageAllocationError, match="all bounded source pages"):
        allocator.allocate("needs-contiguous-slot", 64)


def _scan_truth(allocator: TargetSourcePageAllocator) -> tuple[int, int, int]:
    arena_leases = [
        lease for arena in allocator._arenas for lease in arena.leases.values()
    ]
    assert sorted(lease.lease_id for lease in arena_leases) == sorted(
        allocator._leases
    )
    return (
        len(allocator._leases),
        sum(count for arena in allocator._arenas for _, count in arena.free_extents),
        sum(lease.internal_fragmentation_rows for lease in arena_leases),
    )


def _assert_cache_matches_scan(allocator: TargetSourcePageAllocator) -> None:
    cached = (
        allocator.active_lease_count,
        allocator.free_page_count,
        allocator.internal_fragmentation_rows,
    )
    assert cached == _scan_truth(allocator)


def test_fixed_seed_transitions_keep_cached_counts_exact() -> None:
    allocator = TargetSourcePageAllocator(slot_count=3, slot_rows=128, page_rows=32)
    rng = random.Random(20261121)
    active: list[SourceExtentLease] = []
    for step in range(70):
        if active and (len(active) >= 6 or rng.random() < 0.45):
            lease = active.pop(rng.randrange(len(active)))
            allocator.release(
                lease.lease_id,
                slot_id=lease.slot_id,
                slot_generation=lease.slot_generation,
            )
        else:
            try:
                active.append(
                    allocator.allocate(
                        f"lease-{step}", rng.choice((1, 16, 32, 48, 64))
                    )
                )
            except SourcePageAllocationError:
                pass
        _assert_cache_matches_scan(allocator)

    for lease in active:
        allocator.release(lease.lease_id)
    assert allocator.active_lease_count == 0
    assert allocator.free_page_count == allocator.total_page_count
    assert allocator.internal_fragmentation_rows == 0
    for arena in allocator._arenas:
        assert arena.free_extents == [(0, allocator.pages_per_slot)]
        assert not arena.leases


def test_failed_operations_leave_cached_counts_unchanged() -> None:
    allocator = TargetSourcePageAllocator(slot_count=1, slot_rows=128, page_rows=32)
    for index in range(4):
        allocator.allocate(f"page-{index}", 32)
    before = _scan_truth(allocator)
    with pytest.raises(SourcePageAllocationError, match="all bounded source pages"):
        allocator.allocate("over-full", 32)
    assert _scan_truth(allocator) == before
    _assert_cache_matches_scan(allocator)

    allocator.release("page-1")
    before = _scan_truth(allocator)
    with pytest.raises(SourcePageAllocationError, match="all bounded source pages"):
        allocator.allocate("fragmented", 64)
    assert _scan_truth(allocator) == before
    _assert_cache_matches_scan(allocator)

    live = allocator.get("page-2")
    assert live is not None
    with pytest.raises(SourcePageAllocationError, match="is active"):
        allocator.allocate("page-2", 1)
    with pytest.raises(SourcePageAllocationError, match="unknown source lease"):
        allocator.release("missing")
    with pytest.raises(SourcePageAllocationError, match="stale source slot id"):
        allocator.release("page-2", slot_id=live.slot_id + 1)
    with pytest.raises(SourcePageAllocationError, match="stale source slot generation"):
        allocator.release(
            "page-2",
            slot_id=live.slot_id,
            slot_generation=live.slot_generation + 1,
        )
    assert _scan_truth(allocator) == before
    _assert_cache_matches_scan(allocator)


class _CountingList(list):
    """List wrapper that makes iteration of an extent list observable."""

    iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


def _scalar_stats(allocator: TargetSourcePageAllocator) -> tuple[int, ...]:
    return (
        allocator.active_lease_count,
        allocator.free_page_count,
        allocator.internal_fragmentation_rows,
        allocator.allocation_count,
        allocator.failure_count,
        allocator.requested_rows_total,
        allocator.allocated_rows_total,
        allocator.internal_fragmentation_rows_total,
        allocator.active_lease_count_high_water,
        allocator.leased_page_count_high_water,
    )


def test_scalar_stats_track_success_capacity_failure_and_release() -> None:
    allocator = TargetSourcePageAllocator(slot_count=1, slot_rows=64, page_rows=32)

    lease = allocator.allocate("kept", 48)
    assert _scalar_stats(allocator) == (1, 0, 16, 1, 0, 48, 64, 16, 1, 2)

    # Invalid, duplicate, and stale operations fail before the capacity search.
    with pytest.raises(SourcePageAllocationError, match="is active"):
        allocator.allocate("kept", 1)
    with pytest.raises(SourcePageAllocationError, match="fit inside"):
        allocator.allocate("too-big", 65)
    with pytest.raises(SourcePageAllocationError, match="unknown"):
        allocator.release("missing")
    with pytest.raises(SourcePageAllocationError, match="stale source slot id"):
        allocator.release("kept", slot_id=lease.slot_id + 1)
    assert allocator.failure_count == 0

    with pytest.raises(SourcePageAllocationError, match="all bounded"):
        allocator.allocate("overflow", 1)
    assert _scalar_stats(allocator) == (1, 0, 16, 1, 1, 48, 64, 16, 1, 2)

    # Release restores capacity but never rewinds cumulative/high-water marks.
    allocator.release("kept")
    assert _scalar_stats(allocator) == (0, 2, 0, 1, 1, 48, 64, 16, 1, 2)


def test_scalar_stats_reads_do_not_iterate_extents() -> None:
    allocator = TargetSourcePageAllocator(slot_count=2, slot_rows=128, page_rows=32)
    allocator.allocate("a", 48)
    for arena in allocator._arenas:
        arena.free_extents = _CountingList(arena.free_extents)
    before = tuple(arena.free_extents.iterations for arena in allocator._arenas)

    assert _scalar_stats(allocator) == (1, 6, 16, 1, 0, 48, 64, 16, 1, 2)

    assert tuple(arena.free_extents.iterations for arena in allocator._arenas) == before
