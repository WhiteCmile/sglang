"""Page-granular suballocator for Target feature-source memory.

This module deliberately has no CUDA dependency.  It models the ownership
part of a future ``TargetFeatureSourcePool`` implementation: a fixed set of
registered physical slots is divided into fixed-size pages and a logical
source lease is mapped to one contiguous page extent.  A physical slot keeps
one generation for the whole interval in which any of its extents are live;
the generation advances only after the slot drains completely.

Keeping an allocation inside one physical slot is important for the current
DT wire protocol.  Draft derives the legal source address range from
``(source_slot_id, slot_nbytes)`` and therefore cannot consume an arbitrary
virtual-to-physical scatter list without a protocol change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class SourcePageAllocationError(RuntimeError):
    """Raised when a lease is invalid or the bounded page pool is full."""


@dataclass(frozen=True, slots=True)
class SourceExtentLease:
    """Logical lease mapped to a contiguous physical page extent."""

    lease_id: str
    slot_id: int
    slot_generation: int
    page_start: int
    page_count: int
    requested_rows: int
    page_rows: int

    @property
    def row_offset(self) -> int:
        return self.page_start * self.page_rows

    @property
    def row_capacity(self) -> int:
        return self.page_count * self.page_rows

    @property
    def internal_fragmentation_rows(self) -> int:
        return self.row_capacity - self.requested_rows


@dataclass(slots=True)
class _PageArena:
    generation: int = 0
    free_extents: list[tuple[int, int]] = field(default_factory=list)
    leases: dict[str, SourceExtentLease] = field(default_factory=dict)


class TargetSourcePageAllocator:
    """Best-fit, coalescing extent allocator over fixed-size source pages.

    The allocator never spans a lease across physical slots.  This gives the
    existing Draft protocol one real ``source_slot_id`` plus an in-slot byte
    offset, while allowing unrelated waves to share the otherwise mostly
    empty 8K-row Target slot.
    """

    def __init__(self, *, slot_count: int, slot_rows: int, page_rows: int) -> None:
        if slot_count <= 0 or slot_rows <= 0 or page_rows <= 0:
            raise ValueError("source page-pool dimensions must be positive")
        if slot_rows % page_rows:
            raise ValueError("slot_rows must be an exact multiple of page_rows")
        self.slot_count = int(slot_count)
        self.slot_rows = int(slot_rows)
        self.page_rows = int(page_rows)
        self.pages_per_slot = self.slot_rows // self.page_rows
        self._arenas = [
            _PageArena(free_extents=[(0, self.pages_per_slot)])
            for _ in range(self.slot_count)
        ]
        self._leases: dict[str, SourceExtentLease] = {}
        # O(1) mirrors of the arena/lease bookkeeping.  They are updated only
        # after the corresponding arena and ``_leases`` mutation commits, so
        # rejected or stale operations leave them untouched.
        self._free_page_count = self.slot_count * self.pages_per_slot
        self._active_lease_count = 0
        self._internal_fragmentation_rows = 0

    def allocate(self, lease_id: str, requested_rows: int) -> SourceExtentLease:
        """Map ``lease_id`` to the smallest sufficient free page extent."""

        if not lease_id:
            raise ValueError("lease_id must not be empty")
        if lease_id in self._leases:
            raise SourcePageAllocationError(f"source lease {lease_id!r} is active")
        if requested_rows <= 0 or requested_rows > self.slot_rows:
            raise SourcePageAllocationError(
                "requested rows must fit inside one physical source slot"
            )
        pages = (int(requested_rows) + self.page_rows - 1) // self.page_rows

        # Best fit bounds external fragmentation without adding buddy-style
        # power-of-two rounding on top of page rounding.  Ties prefer the
        # lowest physical address for deterministic traces.
        choice: tuple[int, int, int, int] | None = None
        for slot_id, arena in enumerate(self._arenas):
            for extent_index, (page_start, page_count) in enumerate(
                arena.free_extents
            ):
                if page_count < pages:
                    continue
                candidate = (page_count, slot_id, page_start, extent_index)
                if choice is None or candidate < choice:
                    choice = candidate
        if choice is None:
            raise SourcePageAllocationError("all bounded source pages are leased")

        _free_count, slot_id, page_start, extent_index = choice
        arena = self._arenas[slot_id]
        free_start, free_count = arena.free_extents[extent_index]
        if not arena.leases:
            # A generation covers an entire non-empty epoch of this physical
            # slot.  Sharing the generation is safe only while the slot never
            # returns to the completely-free state.
            arena.generation += 1
        remaining = free_count - pages
        if remaining:
            arena.free_extents[extent_index] = (free_start + pages, remaining)
        else:
            arena.free_extents.pop(extent_index)
        lease = SourceExtentLease(
            lease_id=lease_id,
            slot_id=slot_id,
            slot_generation=arena.generation,
            page_start=page_start,
            page_count=pages,
            requested_rows=int(requested_rows),
            page_rows=self.page_rows,
        )
        arena.leases[lease_id] = lease
        self._leases[lease_id] = lease
        self._free_page_count -= pages
        self._active_lease_count += 1
        self._internal_fragmentation_rows += lease.internal_fragmentation_rows
        return lease

    def release(
        self,
        lease_id: str,
        *,
        slot_id: int | None = None,
        slot_generation: int | None = None,
    ) -> SourceExtentLease:
        """Release one lease and coalesce adjacent free page extents."""

        lease = self._leases.get(lease_id)
        if lease is None:
            raise SourcePageAllocationError(f"unknown source lease {lease_id!r}")
        if slot_id is not None and int(slot_id) != lease.slot_id:
            raise SourcePageAllocationError("stale source slot id")
        if (
            slot_generation is not None
            and int(slot_generation) != lease.slot_generation
        ):
            raise SourcePageAllocationError("stale source slot generation")

        arena = self._arenas[lease.slot_id]
        if arena.generation != lease.slot_generation:
            raise SourcePageAllocationError("source slot epoch changed while active")
        del arena.leases[lease_id]
        del self._leases[lease_id]
        arena.free_extents.append((lease.page_start, lease.page_count))
        arena.free_extents.sort()
        arena.free_extents = self._coalesce(arena.free_extents)
        self._free_page_count += lease.page_count
        self._active_lease_count -= 1
        self._internal_fragmentation_rows -= lease.internal_fragmentation_rows
        return lease

    def get(self, lease_id: str) -> SourceExtentLease | None:
        return self._leases.get(lease_id)

    @property
    def active_lease_count(self) -> int:
        return self._active_lease_count

    @property
    def total_page_count(self) -> int:
        return self.slot_count * self.pages_per_slot

    @property
    def free_page_count(self) -> int:
        return self._free_page_count

    @property
    def largest_free_extent_pages(self) -> int:
        return max(
            (
                page_count
                for arena in self._arenas
                for _page_start, page_count in arena.free_extents
            ),
            default=0,
        )

    @property
    def internal_fragmentation_rows(self) -> int:
        return self._internal_fragmentation_rows

    @staticmethod
    def _coalesce(extents: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, count in extents:
            if count <= 0:
                raise SourcePageAllocationError("free extent must be positive")
            if merged and merged[-1][0] + merged[-1][1] == start:
                prev_start, prev_count = merged[-1]
                merged[-1] = (prev_start, prev_count + count)
            else:
                merged.append((start, count))
        return merged
