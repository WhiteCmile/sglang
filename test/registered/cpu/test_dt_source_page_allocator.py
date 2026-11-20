from __future__ import annotations

import pytest
from sglang.srt.speculative.dt_source_page_allocator import (
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

