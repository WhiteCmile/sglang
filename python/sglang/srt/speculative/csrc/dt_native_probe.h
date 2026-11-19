// D0 per-hop instrumentation of the native Draft workers (issue #269).
//
// A probe exists only while profiling is on; a worker without one pays a
// null check per site.  Each probe has exactly one writer, its worker
// thread, and one reader, the Draft scheduler thread, which drains it into
// the profile directory (dt_direct_ring.NativeProbe):
//
//   stamps      SPSC ring of fixed records {t_ns, kind, ring, seq, arg};
//               head is written by the worker, tail by the reader, each with
//               a release store read by an acquire load.  A full ring drops
//               the stamp and counts it, never blocking the worker.
//   histograms  per timed driver call, log-linear duration buckets (8 per
//               octave).  Single writer; the reader's relaxed loads may see
//               a count one update old, which a cumulative dump tolerates.
//
// Every timestamp is CLOCK_MONOTONIC ns, the clock of time.monotonic_ns().
// Kind and call names, and bucket bounds, are exported (dt_probe_*_name,
// dt_probe_bucket_floor) so the Python side never mirrors these tables.

#pragma once

#include <stdint.h>
#include <time.h>

#include <atomic>
#include <vector>

namespace dt_probe {

#define DT_PROBE_KINDS(X)                                              \
  X(RECORD_SEEN, "record_seen")             /* entry/arrival first seen */ \
  X(MAILBOX_PUBLISHED, "mailbox_published") /* mailbox seq tag stored */   \
  X(PLACEMENT, "placement")                 /* landing space placed; arg = k<<40|offset */ \
  X(COPY_ENQUEUED, "copy_enqueued")         /* event wait + copy queued */  \
  X(INPUT_READY_ISSUED, "input_ready_issued") /* INPUT_READY write queued/stored */ \
  X(READ_POSTED, "read_posted")             /* cross-host RDMA READ posted */ \
  X(READ_COMPLETED, "read_completed")       /* its completion reaped */      \
  X(JOB_TAKEN, "job_taken")                 /* publish job dequeued; seq = ticket */ \
  X(JOB_ISSUED, "job_issued")               /* its event re-record issued */

#define DT_PROBE_CALLS(X)                            \
  X(IPC_OPEN_EVENT, "cuIpcOpenEventHandle")          \
  X(STREAM_WAIT_EVENT, "cuStreamWaitEvent")          \
  X(MEMCPY_ASYNC, "cuMemcpyAsync")                   \
  X(WRITE_VALUE32, "cuStreamWriteValue32")           \
  X(EVENT_DESTROY, "cuEventDestroy")                 \
  X(EVENT_RECORD, "cuEventRecord")                   \
  X(LAUNCH_HOST_FUNC, "cuLaunchHostFunc")            \
  X(FLUSH_RDMA_WRITES, "cuFlushGPUDirectRDMAWrites") \
  X(RDMA_POST_READ, "ibv_post_send(READ)")

#define DT_PROBE_ENUM(id, name) id,
enum Kind : uint32_t { KIND_NONE = 0, DT_PROBE_KINDS(DT_PROBE_ENUM) KIND_COUNT };
enum Call : uint32_t { DT_PROBE_CALLS(DT_PROBE_ENUM) CALL_COUNT };
#undef DT_PROBE_ENUM

constexpr uint32_t kSubBuckets = 8;  // per octave
constexpr uint32_t kBuckets = 512;   // covers every u64 duration

struct Stamp {
  uint64_t t_ns;
  uint32_t kind;
  int32_t ring;
  uint64_t seq;
  uint64_t arg;
};
static_assert(sizeof(Stamp) == 32, "stamp record is four u64 words");

struct Probe {
  explicit Probe(uint64_t capacity) : stamps(capacity) {}
  std::vector<Stamp> stamps;
  std::atomic<uint64_t> head{0};  // writer
  std::atomic<uint64_t> tail{0};  // reader
  std::atomic<uint64_t> dropped{0};
  std::atomic<uint64_t> hist[CALL_COUNT][kBuckets] = {};
};

inline uint64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

inline uint32_t bucket(uint64_t ns) {
  if (ns < kSubBuckets) return (uint32_t)ns;
  const uint32_t msb = 63 - __builtin_clzll(ns);
  return (msb - 2) * kSubBuckets + (uint32_t)((ns >> (msb - 3)) & (kSubBuckets - 1));
}

inline uint64_t bucket_floor(uint32_t index) {
  if (index < kSubBuckets) return index;
  const uint32_t msb = index / kSubBuckets + 2;
  return (uint64_t)(kSubBuckets + index % kSubBuckets) << (msb - 3);
}

inline void stamp(Probe* p, Kind kind, int ring, uint64_t seq, uint64_t arg = 0) {
  if (p == nullptr) return;
  const uint64_t head = p->head.load(std::memory_order_relaxed);
  if (head - p->tail.load(std::memory_order_acquire) >= p->stamps.size()) {
    p->dropped.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  p->stamps[head % p->stamps.size()] = Stamp{now_ns(), kind, ring, seq, arg};
  p->head.store(head + 1, std::memory_order_release);
}

// Run `call` (a driver call), timing it into `which` when probed.
template <class F>
inline auto timed(Probe* p, Call which, F&& call) -> decltype(call()) {
  if (p == nullptr) return call();
  const uint64_t started = now_ns();
  auto rc = call();
  p->hist[which][bucket(now_ns() - started)].fetch_add(1, std::memory_order_relaxed);
  return rc;
}

}  // namespace dt_probe

// Reader side, defined in every library that includes this header (one
// translation unit each); ctypes binds them per library, so equal names in
// two libraries never collide.
extern "C" {

__attribute__((visibility("default"))) void* dt_probe_create(uint64_t capacity) {
  return capacity ? new dt_probe::Probe(capacity) : nullptr;
}

__attribute__((visibility("default"))) void dt_probe_destroy(void* probe) {
  delete static_cast<dt_probe::Probe*>(probe);
}

// Move up to `max` stamps (4 u64 words each) into `out`; returns the count.
__attribute__((visibility("default"))) uint64_t dt_probe_drain(void* probe, uint64_t* out,
                                                                    uint64_t max) {
  auto* p = static_cast<dt_probe::Probe*>(probe);
  const uint64_t tail = p->tail.load(std::memory_order_relaxed);
  const uint64_t head = p->head.load(std::memory_order_acquire);
  const uint64_t n = head - tail < max ? head - tail : max;
  for (uint64_t i = 0; i < n; ++i)
    __builtin_memcpy(out + 4 * i, &p->stamps[(tail + i) % p->stamps.size()],
                     sizeof(dt_probe::Stamp));
  p->tail.store(tail + n, std::memory_order_release);
  return n;
}

__attribute__((visibility("default"))) uint64_t dt_probe_dropped(void* probe) {
  return static_cast<dt_probe::Probe*>(probe)->dropped.load(std::memory_order_relaxed);
}

// Copy the cumulative histograms, CALL_COUNT rows of kBuckets u64 counts.
__attribute__((visibility("default"))) void dt_probe_histograms(void* probe,
                                                                     uint64_t* out) {
  auto* p = static_cast<dt_probe::Probe*>(probe);
  for (uint32_t c = 0; c < dt_probe::CALL_COUNT; ++c)
    for (uint32_t b = 0; b < dt_probe::kBuckets; ++b)
      out[c * dt_probe::kBuckets + b] = p->hist[c][b].load(std::memory_order_relaxed);
}

__attribute__((visibility("default"))) uint32_t dt_probe_call_count() {
  return dt_probe::CALL_COUNT;
}

__attribute__((visibility("default"))) uint32_t dt_probe_bucket_count() {
  return dt_probe::kBuckets;
}

__attribute__((visibility("default"))) uint64_t dt_probe_bucket_floor(uint32_t index) {
  return dt_probe::bucket_floor(index);
}

// Names by index; nullptr past the end (kind 0 is unnamed).
__attribute__((visibility("default"))) const char* dt_probe_kind_name(uint32_t kind) {
#define DT_PROBE_NAME(id, name) name,
  static const char* const names[] = {nullptr, DT_PROBE_KINDS(DT_PROBE_NAME)};
  return kind < dt_probe::KIND_COUNT ? names[kind] : nullptr;
}

__attribute__((visibility("default"))) const char* dt_probe_call_name(uint32_t call) {
  static const char* const names[] = {DT_PROBE_CALLS(DT_PROBE_NAME)};
#undef DT_PROBE_NAME
  return call < dt_probe::CALL_COUNT ? names[call] : nullptr;
}

}  // extern "C"
