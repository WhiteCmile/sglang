// DT V2 Target-side completion-record observer.
//
// One native thread per Target candidate arena scans the /dev/shm
// completion-record mapping (and the legacy READY-record mapping when one
// exists), applying the same cheap prefilter the Python poll used to run on
// the scheduler thread: the slot's armed reservation nonce (seq word) and
// generation.  Every newly matching record is appended, with the moment it
// was seen, to a lock-free SPSC ring that the scheduler thread drains in
// poll().  The heavy per-record validation (transfer hash, producer tag,
// slot state machine) stays in Python; this thread only records arrivals, so
// a wave's last member is observed while the scheduler is still busy, not at
// the next poll.  No GIL, no CUDA: the library links only pthread.
//
// Record word indices (completion seq/meta, READY generation) are passed in
// by the Python owner, which single-sources the layout constants.  The
// expectation arrays (_record_seq_expected/_record_generation) are numpy
// memory owned by the arena, written under its lock; aligned u32/u64 loads
// cannot tear, a stale mismatch self-heals on the next pass, and the Python
// drain revalidates every entry under the lock before consuming it.

#include <stdint.h>
#include <time.h>

#include <atomic>
#include <new>
#include <thread>

namespace {

// Ring entry, one per observed record.  `value` is the matched seq (kind 1)
// or generation (kind 2).
struct Entry {
  uint64_t kind;
  uint64_t slot;
  uint64_t value;
  uint64_t observed_ns;
};
static_assert(sizeof(Entry) == 32, "observer ring entry must be 32 bytes");

constexpr uint64_t kKindCompletion = 1;
constexpr uint64_t kKindReady = 2;
// One ring per record kind so each Python poll method drains only its own
// arrivals.  The per-slot dedup arrays bound pending entries per kind well
// below this; overflow only sets `dropped` and the Python side re-runs its
// legacy full scan for that poll.
constexpr uint64_t kRingCapacityFactor = 4;
// Idle/backoff pause range for the scan loop (see run()).
constexpr uint64_t kBackoffMinNs = 20000;     // 20 us
constexpr uint64_t kBackoffMaxNs = 1000000;   // 1 ms

uint64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

struct Observer {
  uint32_t slot_count = 0;
  // Word indices inside one completion record (u64 words) and one READY
  // record (u32 words), owned by the Python caller's layout constants.
  uint32_t completion_seq_word = 0;   // reservation nonce, released last
  uint32_t completion_meta_word = 0;  // generation (low u32), length (high)
  uint32_t ready_generation_word = 0;
  // Completion records: slot s lives at completion + s * completion_stride.
  volatile const uint64_t* completion = nullptr;
  uint64_t completion_stride = 0;
  // Legacy READY records (u32 words); nullptr when the arena has none.
  volatile const uint32_t* ready = nullptr;
  uint64_t ready_stride = 0;
  // numpy-owned expectation arrays; index 0 means "not waiting".
  volatile const uint64_t* seq_expected = nullptr;
  volatile const uint32_t* generation = nullptr;
  // Per-slot dedup: the last reported match, so a record that keeps matching
  // enqueues exactly once until the expectation moves on.
  uint64_t* reported_seq = nullptr;
  uint32_t* reported_gen = nullptr;
  Entry* rings[2] = {nullptr, nullptr};
  uint64_t ring_capacity = 0;
  // head: read cursor (Python drain), tail: write cursor (this thread).
  std::atomic<uint64_t> head[2];
  std::atomic<uint64_t> tail[2];
  std::atomic<uint64_t> dropped{0};
  std::atomic<bool> stop{false};
  std::thread thread;
};

bool push(Observer* o, int kind, uint32_t slot, uint64_t value) {
  uint64_t tail = o->tail[kind].load(std::memory_order_relaxed);
  uint64_t head = o->head[kind].load(std::memory_order_acquire);
  if (tail - head >= o->ring_capacity) return false;
  Entry& e = o->rings[kind][tail % o->ring_capacity];
  e.kind = kind == 0 ? kKindCompletion : kKindReady;
  e.slot = slot;
  e.value = value;
  e.observed_ns = now_ns();
  o->tail[kind].store(tail + 1, std::memory_order_release);
  return true;
}

void run(Observer* o) {
  // Idle backoff: a pass that observed nothing doubles the pause up to the
  // cap, so an idle arena sleeps instead of pinning a core; one new arrival
  // resets the pause to the floor.  The floor also caps the scan rate while
  // records stream in, at a fraction of a millisecond of added observation
  // latency.
  uint64_t backoff_ns = kBackoffMinNs;
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
      // acquire before reading the rest of the record or publishing the
      // arrival (the same fence dt_ring_worker.cpp places after its
      // publication-word load).
      std::atomic_thread_fence(std::memory_order_acquire);
      if ((uint32_t)rec[o->completion_meta_word] != o->generation[s]) continue;
      if (push(o, 0, s, seq)) {
        o->reported_seq[s] = seq;
        progressed = true;
      } else {
        o->dropped.fetch_add(1, std::memory_order_relaxed);
      }
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
        if (push(o, 1, s, gen)) {
          o->reported_gen[s] = gen;
          progressed = true;
        } else {
          o->dropped.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
    if (progressed)
      backoff_ns = kBackoffMinNs;
    else if (backoff_ns < kBackoffMaxNs)
      backoff_ns = backoff_ns * 2 > kBackoffMaxNs ? kBackoffMaxNs : backoff_ns * 2;
    timespec ts;
    ts.tv_sec = (time_t)(backoff_ns / 1000000000ull);
    ts.tv_nsec = (long)(backoff_ns % 1000000000ull);
    clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, nullptr);
  }
}

}  // namespace

extern "C" {

// Returns an opaque observer or nullptr.  `ready_base` may be 0 when the
// arena has no READY-record mapping.  The expectation arrays and both record
// mappings must outlive the observer (the Python owner destroys it before
// unmapping).
void* dt_record_observer_create(uint32_t slot_count, uint64_t completion_base,
                                uint64_t completion_stride, uint64_t ready_base,
                                uint64_t ready_stride, uint64_t seq_expected_base,
                                uint64_t generation_base, uint32_t completion_seq_word,
                                uint32_t completion_meta_word,
                                uint32_t ready_generation_word) {
  if (!slot_count || !completion_base || !completion_stride || !seq_expected_base ||
      !generation_base)
    return nullptr;
  Observer* o = new (std::nothrow) Observer();
  if (!o) return nullptr;
  o->slot_count = slot_count;
  o->completion_seq_word = completion_seq_word;
  o->completion_meta_word = completion_meta_word;
  o->ready_generation_word = ready_generation_word;
  o->completion = reinterpret_cast<volatile const uint64_t*>(completion_base);
  o->completion_stride = completion_stride;
  o->ready = ready_base
                 ? reinterpret_cast<volatile const uint32_t*>(ready_base)
                 : nullptr;
  o->ready_stride = ready_stride;
  o->seq_expected = reinterpret_cast<volatile const uint64_t*>(seq_expected_base);
  o->generation = reinterpret_cast<volatile const uint32_t*>(generation_base);
  o->ring_capacity = (uint64_t)slot_count * kRingCapacityFactor;
  o->reported_seq = new (std::nothrow) uint64_t[slot_count]();
  o->reported_gen = new (std::nothrow) uint32_t[slot_count]();
  o->rings[0] = new (std::nothrow) Entry[o->ring_capacity];
  o->rings[1] = new (std::nothrow) Entry[o->ring_capacity];
  if (!o->reported_seq || !o->reported_gen || !o->rings[0] || !o->rings[1]) {
    delete[] o->reported_seq;
    delete[] o->reported_gen;
    delete[] o->rings[0];
    delete[] o->rings[1];
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

// Copies up to `out_entries` pending arrivals of `kind` (1 completion,
// 2 READY) into `out` (4 u64 per entry: kind, slot, value, observed_ns) and
// advances the read cursor; returns how many were copied.
uint64_t dt_record_observer_drain(void* handle, uint64_t kind, uint64_t* out,
                                  uint64_t out_entries) {
  Observer* o = static_cast<Observer*>(handle);
  if (!o || (kind != kKindCompletion && kind != kKindReady) || !out) return 0;
  int k = (int)kind - 1;
  uint64_t head = o->head[k].load(std::memory_order_relaxed);
  uint64_t tail = o->tail[k].load(std::memory_order_acquire);
  uint64_t n = tail - head;
  if (n > out_entries) n = out_entries;
  for (uint64_t i = 0; i < n; ++i) {
    const Entry& e = o->rings[k][(head + i) % o->ring_capacity];
    out[i * 4 + 0] = e.kind;
    out[i * 4 + 1] = e.slot;
    out[i * 4 + 2] = e.value;
    out[i * 4 + 3] = e.observed_ns;
  }
  o->head[k].store(head + n, std::memory_order_release);
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
  delete[] o->rings[0];
  delete[] o->rings[1];
  delete o;
  return 0;
}

}  // extern "C"
