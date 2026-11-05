// DT V2 Target-side completion-record observer.
//
// One native thread per Target candidate arena scans the /dev/shm
// completion-record mapping (and the legacy READY-record mapping when one
// exists), applies the full per-record validation the Python poll used to
// run on the scheduler thread -- the armed reservation nonce (seq word),
// generation, transfer hash, and producer tag -- and classifies each first
// match (ready row, authenticated tombstone, or authentication failure).
// Every validated match is published into a per-slot native READY table,
// payload words first and the seq word released last, exactly like the
// producer's own record publication.  The scheduler thread only takes a
// snapshot of that table in poll(): one native call returns the slots that
// became READY since the last snapshot, and the Python side applies the
// slot state machine (READY/tombstone/quarantine) under the arena lock.
// Records that land while the loop is in select, launch or park therefore
// reach READY without waiting for the next poll's drain.  authorize() kicks
// the observer after arming the expectations so a record that beat its ACK
// is re-checked immediately instead of after the idle backoff.  No GIL, no
// CUDA: the library links only pthread.
//
// Record word indices (completion seq/meta, READY generation, and the
// completion payload words) are passed in by the Python owner, which
// single-sources the layout constants.  The expectation arrays
// (_record_seq_expected/_record_generation/_record_tag/_record_hash_expected)
// are numpy memory owned by the arena, written under its lock; aligned
// u32/u64 loads cannot tear, a stale mismatch self-heals on the next pass,
// and the Python snapshot revalidates every entry under the lock before
// consuming it.

#include <stdint.h>
#include <time.h>

#include <atomic>
#include <new>
#include <thread>

namespace {

// Ring entry for the legacy READY-record kind (kind 2), one per observed
// record.  Completion records no longer pass through a ring: their full
// validation and READY publication happen in the table below.
struct Entry {
  uint64_t kind;
  uint64_t slot;
  uint64_t value;
  uint64_t observed_ns;
};
static_assert(sizeof(Entry) == 32, "observer ring entry must be 32 bytes");

// READY-table entry, one per slot.  The observer thread publishes payload
// words first and the seq word (0 = empty; reservation nonces are never 0)
// with a release fence last, the same publication edge the producer uses;
// the snapshot thread acquire-loads the seq word before reading the
// payload.  kind classifies the validated record the same way the Python
// consume path used to: ready row, authenticated tombstone (valid length
// 0), authentication failure (transfer hash or producer tag mismatch), or
// invalid length (authenticated but neither tombstone nor row width).
struct ReadyEntry {
  uint64_t observed_ns;  // when this record was validated
  uint64_t meta;         // generation (low u32), valid length (high u32)
  uint64_t token;        // root token (low u32), producer tag (high u32)
  uint64_t produced;     // producer globaltimer publication stamp
  uint64_t offset;       // producer globaltimer->realtime offset
  std::atomic<uint64_t> seq;
  uint64_t kind;
};
static_assert(sizeof(ReadyEntry) == 56, "observer ready entry must be 56 bytes");

enum ReadyKind : uint64_t {
  kReadyKindRow = 0,
  kReadyKindTombstone = 1,
  kReadyKindAuthFailed = 2,
  kReadyKindBadLength = 3,
};

constexpr uint64_t kKindReady = 2;
// One ring for the legacy READY-record kind.  The per-slot dedup array
// bounds pending entries well below this; overflow only sets `dropped` and
// the Python side re-runs its legacy full scan for that poll.
constexpr uint64_t kRingCapacityFactor = 4;
// Idle/backoff pause range for the scan loop (see run()).
constexpr uint64_t kBackoffMinNs = 20000;     // 20 us
constexpr uint64_t kBackoffMaxNs = 1000000;   // 1 ms
// Long pauses sleep in slices of this size so an authorize() kick is seen
// within one slice instead of after the whole backed-off pause.
constexpr uint64_t kWakeSliceNs = 64000;      // 64 us

uint64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

struct Observer {
  uint32_t slot_count = 0;
  // Word indices inside one completion record (u64 words) and one READY
  // record (u32 words), owned by the Python caller's layout constants.
  uint32_t completion_seq_word = 0;     // reservation nonce, released last
  uint32_t completion_meta_word = 0;    // generation (low u32), length (high)
  uint32_t completion_hash_word = 0;    // transfer id hash
  uint32_t completion_token_word = 0;   // root token (low), producer tag (high)
  uint32_t completion_produced_word = 0;  // producer globaltimer stamp
  uint32_t completion_offset_word = 0;    // producer globaltimer offset
  uint32_t ready_generation_word = 0;
  uint32_t row_width = 0;
  // Completion records: slot s lives at completion + s * completion_stride.
  volatile const uint64_t* completion = nullptr;
  uint64_t completion_stride = 0;
  // Legacy READY records (u32 words); nullptr when the arena has none.
  volatile const uint32_t* ready = nullptr;
  uint64_t ready_stride = 0;
  // numpy-owned expectation arrays; index 0 means "not waiting".
  volatile const uint64_t* seq_expected = nullptr;
  volatile const uint32_t* generation = nullptr;
  volatile const uint64_t* hash_expected = nullptr;
  volatile const uint32_t* tag_expected = nullptr;
  // Per-slot dedup: the last reported match, so a record that keeps matching
  // enqueues exactly once until the expectation moves on.
  uint64_t* reported_seq = nullptr;
  uint32_t* reported_gen = nullptr;
  // Native READY table (observer publishes) and the snapshot thread's
  // per-slot high-water mark of what it has already returned.
  ReadyEntry* ready_entries = nullptr;
  uint64_t* snapshot_seq = nullptr;
  Entry* rings[1] = {nullptr};
  uint64_t ring_capacity = 0;
  // head: read cursor (Python drain), tail: write cursor (this thread).
  std::atomic<uint64_t> head[1];
  std::atomic<uint64_t> tail[1];
  std::atomic<uint64_t> dropped{0};
  std::atomic<uint64_t> wake_epoch{0};
  std::atomic<bool> stop{false};
  std::thread thread;
};

bool push(Observer* o, uint32_t slot, uint64_t value) {
  uint64_t tail = o->tail[0].load(std::memory_order_relaxed);
  uint64_t head = o->head[0].load(std::memory_order_acquire);
  if (tail - head >= o->ring_capacity) return false;
  Entry& e = o->rings[0][tail % o->ring_capacity];
  e.kind = kKindReady;
  e.slot = slot;
  e.value = value;
  e.observed_ns = now_ns();
  o->tail[0].store(tail + 1, std::memory_order_release);
  return true;
}

// Publish one validated completion record into the slot's READY table:
// payload words first, then the seq word with a release fence, so the
// snapshot thread's acquire of the seq word orders its payload reads.
void publish_ready(Observer* o, uint32_t slot, uint64_t seq, uint64_t kind,
                   uint64_t meta, uint64_t token, uint64_t produced,
                   uint64_t offset) {
  ReadyEntry& e = o->ready_entries[slot];
  e.observed_ns = now_ns();
  e.meta = meta;
  e.token = token;
  e.produced = produced;
  e.offset = offset;
  e.kind = kind;
  std::atomic_thread_fence(std::memory_order_release);
  e.seq.store(seq, std::memory_order_relaxed);
}

void run(Observer* o) {
  // Idle backoff: a pass that observed nothing doubles the pause up to the
  // cap, so an idle arena sleeps instead of pinning a core; one new arrival
  // resets the pause to the floor.  The floor also caps the scan rate while
  // records stream in, at a fraction of a millisecond of added observation
  // latency.  Long pauses sleep in wake-sized slices and abort early when
  // authorize() kicks the epoch, so an armed expectation is re-checked
  // within one slice rather than after the full backoff.
  uint64_t backoff_ns = kBackoffMinNs;
  uint64_t seen_epoch = o->wake_epoch.load(std::memory_order_relaxed);
  while (!o->stop.load(std::memory_order_relaxed)) {
    bool progressed = false;
    for (uint32_t s = 0; s < o->slot_count; ++s) {
      uint64_t expected = o->seq_expected[s];
      if (!expected) continue;
      volatile const uint64_t* rec = reinterpret_cast<volatile const uint64_t*>(
          reinterpret_cast<volatile const uint8_t*>(o->completion) +
          (uint64_t)s * o->completion_stride);
      uint64_t seq = rec[o->completion_seq_word];
      if (seq != expected || o->reported_seq[s] == seq) continue;
      // The GPU publishes the seq word with a system-scope release, so
      // acquire before reading the rest of the record (the same fence
      // dt_ring_worker.cpp places after its publication-word load).
      std::atomic_thread_fence(std::memory_order_acquire);
      uint64_t meta = rec[o->completion_meta_word];
      if ((uint32_t)meta != o->generation[s]) continue;
      // Full validation, in the same order the Python consume used: the
      // transfer hash and producer tag bind the record to this
      // reservation's assignment; a valid length of 0 is the producer's
      // authenticated tombstone; any other length than the row width is an
      // authentication failure.
      uint64_t kind = kReadyKindRow;
      uint64_t hash = rec[o->completion_hash_word];
      uint64_t token = rec[o->completion_token_word];
      uint32_t valid_length = (uint32_t)(meta >> 32);
      if (hash != o->hash_expected[s] || (uint32_t)(token >> 32) != o->tag_expected[s])
        kind = kReadyKindAuthFailed;
      else if (valid_length == 0)
        kind = kReadyKindTombstone;
      else if (valid_length != o->row_width)
        kind = kReadyKindBadLength;
      publish_ready(o, s, seq, kind, meta, token,
                    rec[o->completion_produced_word],
                    rec[o->completion_offset_word]);
      o->reported_seq[s] = seq;
      progressed = true;
    }
    if (o->ready) {
      for (uint32_t s = 0; s < o->slot_count; ++s) {
        uint32_t gen = o->generation[s];
        if (!gen || o->reported_gen[s] == gen) continue;
        volatile const uint32_t* rec = reinterpret_cast<volatile const uint32_t*>(
            reinterpret_cast<volatile const uint8_t*>(o->ready) +
            (uint64_t)s * o->ready_stride);
        if (rec[o->ready_generation_word] != gen) continue;
        std::atomic_thread_fence(std::memory_order_acquire);
        if (push(o, s, gen)) {
          o->reported_gen[s] = gen;
          progressed = true;
        } else {
          o->dropped.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
    if (progressed) {
      backoff_ns = kBackoffMinNs;
    } else if (backoff_ns < kBackoffMaxNs) {
      backoff_ns = backoff_ns * 2 > kBackoffMaxNs ? kBackoffMaxNs : backoff_ns * 2;
    }
    uint64_t slept = 0;
    while (slept < backoff_ns &&
           !o->stop.load(std::memory_order_relaxed)) {
      uint64_t epoch = o->wake_epoch.load(std::memory_order_acquire);
      if (epoch != seen_epoch) {
        seen_epoch = epoch;
        break;
      }
      uint64_t slice = backoff_ns - slept;
      if (slice > kWakeSliceNs) slice = kWakeSliceNs;
      timespec ts;
      ts.tv_sec = (time_t)(slice / 1000000000ull);
      ts.tv_nsec = (long)(slice % 1000000000ull);
      clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, nullptr);
      slept += slice;
    }
  }
}

}  // namespace

extern "C" {

// Returns an opaque observer or nullptr.  `ready_base` may be 0 when the
// arena has no READY-record mapping.  The expectation arrays and both record
// mappings must outlive the observer (the Python owner destroys it before
// unmapping).
void* dt_record_observer_create(
    uint32_t slot_count, uint64_t completion_base, uint64_t completion_stride,
    uint64_t ready_base, uint64_t ready_stride, uint64_t seq_expected_base,
    uint64_t generation_base, uint64_t hash_expected_base,
    uint64_t tag_expected_base, uint32_t row_width, uint32_t completion_seq_word,
    uint32_t completion_meta_word, uint32_t ready_generation_word,
    uint32_t completion_hash_word, uint32_t completion_token_word,
    uint32_t completion_produced_word, uint32_t completion_offset_word) {
  if (!slot_count || !completion_base || !completion_stride ||
      !seq_expected_base || !generation_base || !hash_expected_base ||
      !tag_expected_base || !row_width)
    return nullptr;
  Observer* o = new (std::nothrow) Observer();
  if (!o) return nullptr;
  o->slot_count = slot_count;
  o->completion_seq_word = completion_seq_word;
  o->completion_meta_word = completion_meta_word;
  o->completion_hash_word = completion_hash_word;
  o->completion_token_word = completion_token_word;
  o->completion_produced_word = completion_produced_word;
  o->completion_offset_word = completion_offset_word;
  o->ready_generation_word = ready_generation_word;
  o->row_width = row_width;
  o->completion = reinterpret_cast<volatile const uint64_t*>(completion_base);
  o->completion_stride = completion_stride;
  o->ready = ready_base
                 ? reinterpret_cast<volatile const uint32_t*>(ready_base)
                 : nullptr;
  o->ready_stride = ready_stride;
  o->seq_expected = reinterpret_cast<volatile const uint64_t*>(seq_expected_base);
  o->generation = reinterpret_cast<volatile const uint32_t*>(generation_base);
  o->hash_expected = reinterpret_cast<volatile const uint64_t*>(hash_expected_base);
  o->tag_expected = reinterpret_cast<volatile const uint32_t*>(tag_expected_base);
  o->ring_capacity = (uint64_t)slot_count * kRingCapacityFactor;
  o->reported_seq = new (std::nothrow) uint64_t[slot_count]();
  o->reported_gen = new (std::nothrow) uint32_t[slot_count]();
  o->ready_entries = new (std::nothrow) ReadyEntry[slot_count]();
  o->snapshot_seq = new (std::nothrow) uint64_t[slot_count]();
  o->rings[0] = new (std::nothrow) Entry[o->ring_capacity];
  if (!o->reported_seq || !o->reported_gen || !o->ready_entries ||
      !o->snapshot_seq || !o->rings[0]) {
    delete[] o->reported_seq;
    delete[] o->reported_gen;
    delete[] o->ready_entries;
    delete[] o->snapshot_seq;
    delete[] o->rings[0];
    delete o;
    return nullptr;
  }
  return o;
}

int dt_record_observer_start(void* handle) {
  Observer* o = static_cast<Observer*>(handle);
  if (!o || o->thread.joinable()) return -1;
  try {
    o->thread = std::thread(run, o);
  } catch (...) {
    return -1;
  }
  return 0;
}

// Wakes the scan early: authorize() calls this after arming the record
// expectations so a record that landed before its assignment is re-checked
// within one wake slice instead of after the idle backoff.
int dt_record_observer_kick(void* handle) {
  Observer* o = static_cast<Observer*>(handle);
  if (!o) return -1;
  o->wake_epoch.fetch_add(1, std::memory_order_release);
  return 0;
}

// Copies up to `out_entries` slots of the READY table that became READY
// since the last snapshot into `out` (8 u64 per entry: slot, seq, observed_ns,
// meta, token, produced, offset, kind) and advances the per-slot snapshot
// cursor; returns how many were copied.  `out_entries` >= slot_count drains
// the table fully: the table is per-slot, so entries can never exceed the
// slot count.
uint64_t dt_record_observer_ready_snapshot(void* handle, uint64_t* out,
                                           uint64_t out_entries) {
  Observer* o = static_cast<Observer*>(handle);
  if (!o || !out) return 0;
  uint64_t n = 0;
  for (uint32_t s = 0; s < o->slot_count && n < out_entries; ++s) {
    ReadyEntry& e = o->ready_entries[s];
    // The observer can overwrite a mark only after the slot was freed and
    // rearmed; re-read the seq word after the payload copy and retry when
    // it moved, so a mixed old/new payload is never returned.
    for (;;) {
      uint64_t seq = e.seq.load(std::memory_order_acquire);
      if (!seq || seq == o->snapshot_seq[s]) break;
      std::atomic_thread_fence(std::memory_order_acquire);
      uint64_t observed_ns = e.observed_ns;
      uint64_t meta = e.meta;
      uint64_t token = e.token;
      uint64_t produced = e.produced;
      uint64_t offset = e.offset;
      uint64_t kind = e.kind;
      std::atomic_thread_fence(std::memory_order_acquire);
      if (e.seq.load(std::memory_order_acquire) != seq) continue;
      o->snapshot_seq[s] = seq;
      out[n * 8 + 0] = s;
      out[n * 8 + 1] = seq;
      out[n * 8 + 2] = observed_ns;
      out[n * 8 + 3] = meta;
      out[n * 8 + 4] = token;
      out[n * 8 + 5] = produced;
      out[n * 8 + 6] = offset;
      out[n * 8 + 7] = kind;
      ++n;
      break;
    }
  }
  return n;
}

// Copies up to `out_entries` pending legacy READY-record arrivals into
// `out` (4 u64 per entry: kind, slot, value, observed_ns) and advances the
// read cursor; returns how many were copied.
uint64_t dt_record_observer_drain(void* handle, uint64_t kind, uint64_t* out,
                                  uint64_t out_entries) {
  Observer* o = static_cast<Observer*>(handle);
  if (!o || kind != kKindReady || !out) return 0;
  uint64_t head = o->head[0].load(std::memory_order_relaxed);
  uint64_t tail = o->tail[0].load(std::memory_order_acquire);
  uint64_t n = tail - head;
  if (n > out_entries) n = out_entries;
  for (uint64_t i = 0; i < n; ++i) {
    const Entry& e = o->rings[0][(head + i) % o->ring_capacity];
    out[i * 4 + 0] = e.kind;
    out[i * 4 + 1] = e.slot;
    out[i * 4 + 2] = e.value;
    out[i * 4 + 3] = e.observed_ns;
  }
  o->head[0].store(head + n, std::memory_order_release);
  return n;
}

uint64_t dt_record_observer_dropped(void* handle) {
  Observer* o = static_cast<Observer*>(handle);
  return o ? o->dropped.load(std::memory_order_relaxed) : 0;
}

// Stop the thread and release everything it owns.  The caller must not have
// unmapped the record files or freed the expectation arrays yet.
int dt_record_observer_destroy(void* handle) {
  Observer* o = static_cast<Observer*>(handle);
  if (!o) return 0;
  o->stop.store(true);
  if (o->thread.joinable()) o->thread.join();
  delete[] o->reported_seq;
  delete[] o->reported_gen;
  delete[] o->ready_entries;
  delete[] o->snapshot_seq;
  delete[] o->rings[0];
  delete o;
  return 0;
}

}  // extern "C"
