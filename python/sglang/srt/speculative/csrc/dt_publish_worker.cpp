// DT Draft candidate publish worker (issue #244).
//
// One native thread per Draft process issues the stream work of candidate
// pushes off the Draft scheduler thread: data movement only.  The scheduler
// thread decides everything (which rows, their destinations, every word and
// its order) and hands each push over as one precomputed job; this thread
// issues the job's work in order on its own CUDA stream, without the GIL:
//
//   cuStreamWaitEvent(event)          the scheduler recorded `event` on its
//                                     submitting stream just before the
//                                     handoff, so the job is ordered after
//                                     everything already queued there
//   cuMemcpyAsync per copy            candidate rows -> Target arena rows /
//                                     cross-host mirror rows
//   cuStreamWriteValue32 per word     the publish words, in the given order
//   cuLaunchHostFunc(post, list)      cross-host GPUDirect WRITE post, if any
//   cuEventRecord(event)              the push's completion event
//
// Jobs run in FIFO order on one stream, so pushes keep the order they had on
// the submitting stream.  cuStreamWaitEvent binds to the record current at
// the call, so re-recording the same event afterwards is the completion.
// Until that record is issued the event still reports the scheduler's
// record, which may be complete: the scheduler must not trust query() for a
// ticket that status() does not report as recorded.
//
// The scheduler thread is the only producer and this thread the only
// consumer of a lock-free single-producer/single-consumer ring.  A failed
// driver call breaks the worker for good: that job and every later one
// report the error, and the scheduler's quarantine takes over.  CUDA is
// reached through the driver API (dlopen); the library links only libdl and
// pthread.
//
// Send/completion loop (Draft redesign D8, issue #277, design 7.1-7.2,
// DT_DRAFT_SEND_LOOP=1).  The same ring also carries *batch descriptors*:
// the scheduler hands one over per submitted compute batch (and one per
// fenced landing release), and this thread watches their completion events
// in FIFO order -- they are recorded on the one compute stream, so the
// order they were queued in is the order they complete in.  Descriptors do
// not block push jobs: a descriptor is moved off the ring into this
// thread's own FIFO, and the head of that FIFO is polled between ring
// drains.  At a batch's GPU done this thread, in order:
//
//   1. emits the DONE record (batch seq, host-observed gpu_done_ns, CUDA
//      elapsed materialize_start -> materialize_done -> proposal_done);
//   2. publishes done_seq (release store), so in-flight is
//      frozen_seq - done_seq, two single-writer monotonic counters;
//   3. sets the DONE wake bit, waking the scheduler;
//   4. pushes the batch's landing release records to the ingress worker.
//
// The record is published before done_seq and the wake, so a scheduler that
// sees the counter move always finds the record (design 4.3: the scheduler
// is the only reader of both).  Nothing here touches Python (R4).
//
// Pushes (D9, design 7.2 step 3 to 7.4).  A descriptor also lists the batch's
// candidate rows that the scheduler could not pre-queue at submit: same-host
// v2 rows whose destination was not in the region yet (LATE_LOCAL) and
// cross-host rows (CROSS_HOST).  At the GPU done this thread resolves each
// LATE_LOCAL row from the destination region (seqlock), builds its copy and
// publication words natively, copies the CROSS_HOST rows into their push
// leg's mirror, and issues all of it on a second stream (`late`), so it never
// queues behind a later batch's pre-queued push waiting on that batch's
// proposal.  Once that work completes it posts the cross-host WRITEs itself
// (no CUDA host function).  A row whose destination is still absent is
// held and re-read every pass until its deadline.  Every push job's
// completion (the pre-queued ones on `stream` too) becomes a PUSH_DONE
// record, so the scheduler polls no push event.

#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <sys/prctl.h>
#include <stdint.h>
#include <string.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "dt_native_probe.h"

namespace {

using dt_probe::timed;

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUstream_st* CUstream;
typedef struct CUevent_st* CUevent;
typedef unsigned long long CUdeviceptr;
typedef void (*CUhostFn)(void*);

struct Api {
  CUresult (*cuInit)(unsigned);
  CUresult (*cuDeviceGet)(CUdevice*, int);
  CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
  CUresult (*cuDevicePrimaryCtxRelease)(CUdevice);
  CUresult (*cuCtxSetCurrent)(CUcontext);
  CUresult (*cuStreamCreateWithPriority)(CUstream*, unsigned, int);
  CUresult (*cuStreamDestroy)(CUstream);
  CUresult (*cuStreamSynchronize)(CUstream);
  CUresult (*cuStreamWaitEvent)(CUstream, CUevent, unsigned);
  CUresult (*cuMemcpyAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
  CUresult (*cuStreamWriteValue32)(CUstream, CUdeviceptr, uint32_t, unsigned);
  CUresult (*cuLaunchHostFunc)(CUstream, CUhostFn, void*);
  CUresult (*cuEventRecord)(CUevent, CUstream);
  CUresult (*cuEventQuery)(CUevent);
  CUresult (*cuEventElapsedTime)(float*, CUevent, CUevent);
  CUresult (*cuEventCreate)(CUevent*, unsigned);
  CUresult (*cuEventDestroy)(CUevent);
};

constexpr CUresult kNotReady = 600;  // CUDA_ERROR_NOT_READY

bool load_api(Api* api) {
  void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
  if (!h) h = dlopen("libcuda.so.1", RTLD_NOW);
  if (!h) return false;
#define LOAD(field, name) \
  api->field = reinterpret_cast<decltype(api->field)>(dlsym(h, name)); \
  if (!api->field) return false;
  LOAD(cuInit, "cuInit");
  LOAD(cuDeviceGet, "cuDeviceGet");
  LOAD(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
  LOAD(cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2");
  LOAD(cuCtxSetCurrent, "cuCtxSetCurrent");
  LOAD(cuStreamCreateWithPriority, "cuStreamCreateWithPriority");
  LOAD(cuStreamDestroy, "cuStreamDestroy_v2");
  LOAD(cuStreamSynchronize, "cuStreamSynchronize");
  LOAD(cuStreamWaitEvent, "cuStreamWaitEvent");
  LOAD(cuMemcpyAsync, "cuMemcpyAsync");
  LOAD(cuStreamWriteValue32, "cuStreamWriteValue32_v2");
  LOAD(cuLaunchHostFunc, "cuLaunchHostFunc");
  LOAD(cuEventRecord, "cuEventRecord");
  LOAD(cuEventQuery, "cuEventQuery");
  LOAD(cuEventElapsedTime, "cuEventElapsedTime");
  LOAD(cuEventCreate, "cuEventCreate");
  LOAD(cuEventDestroy, "cuEventDestroy_v2");
#undef LOAD
  return true;
}

// Job layout (u64 words), written by the scheduler thread:
//   [0] CUevent  [1] copy count  [2] word count  [3] host fn  [4] host fn arg
//   then (destination, source, bytes) per copy, then (address, value) per word.
constexpr uint64_t kHeaderWords = 5;
constexpr uint32_t kRingSlots = 1024;
// Spin this long for the next job before sleeping on the condition variable.
constexpr auto kSpin = std::chrono::microseconds(50);

// Batch descriptor words (D8), written by the scheduler thread:
//   [0] done seq (the batch's frozen_seq; 0: a release-only fence)
//   [1] CUevent to wait on (the batch's completion event, or the fence)
//   [2] materialize_start, [3] materialize_done, [4] proposal_done event
//   [5] landing release count, then (buffer k, generation, placements) each.
constexpr uint64_t kBatchHeaderWords = 6;
// DONE records: {seq, gpu_done_ns, materialize_ns, proposal_ns, error}.  The
// scheduler drains them on every wake; sized far above the two batches plus
// the fences one DRAIN can produce, and overflow is fail-stop (R7).
constexpr uint64_t kDoneWords = 5;
constexpr uint32_t kDoneSlots = 4096;

// D9 push section, after the landing releases: [0] push row count P, [1] post
// fn, [2] post arg (the cross-host groups' native post, dt_rdma.CandidatePost;
// 0: none), then kPushRowWords per row:
//   [0] class  [1] candidate row (device)  [2] row bytes
//   CROSS_HOST: [3] its mirror row (device), the rest 0
//   LATE_LOCAL: [3] the record's destination-region entry (host)  [4] record
//     seq  [5] request position  [6] arena data base  [7] data allocation
//     length  [8] metadata base  [9] metadata allocation length
//     [10] completion records base (device)  [11] completion record slots
//     [12] transfer hash  [13] root token  [14] deadline (CLOCK_MONOTONIC ns)
constexpr uint64_t kPushHeaderWords = 3;
constexpr uint64_t kPushRowWords = 15;
enum PushClass : uint64_t { PUSH_LATE_LOCAL = 1, PUSH_CROSS_HOST = 2 };
// Layout constants the scheduler passes once (enable_batches), so the region
// and arena geometry has one source (dt_direct_ring / dt_candidate_p2p).
enum PushConst : uint32_t {
  K_DEST_SEQ_AT, K_DEST_DATA_AT, K_DEST_META_AT, K_DEST_ENTRY_BYTES,
  K_METADATA_WORDS, K_RECORD_BYTES, K_PRODUCER_TAG, K_PUSH_CONSTS
};
// PUSH_DONE records {kind, id, row, outcome, issued_ns, done_ns}: one per push
// job the scheduler queued (id = ticket), one per descriptor push row (id =
// done seq, row = its index).  issued_ns is when its stream work was issued.
constexpr uint64_t kPushDoneWords = 6;
constexpr uint32_t kPushDoneSlots = 8192;
enum PushDoneKind : uint64_t { PUSH_DONE_JOB = 1, PUSH_DONE_ROW = 2 };
enum PushOutcome : uint64_t { PUSHED = 0, CANCELLED = 1, EXPIRED = 2, FAILED = 3 };
// One row's publication: 5 metadata words, then 12 completion-record words.
constexpr uint64_t kPublicationWords = 17;
// Bounded wait (controller, from #277: the D8 spin cost ~32% of a core).  A
// batch completes no earlier than kGuardNs before the shortest of its class's
// last kWindow spans (start -> GPU done); until then the thread sleeps on its
// condition variable (a queued job still wakes it at once), re-reading held
// rows every kPollSleep.
constexpr uint64_t kGuardNs = 200000;
constexpr auto kPollSleep = std::chrono::microseconds(50);
constexpr int kWindow = 8;

enum JobKind : uint32_t { JOB_PUSH = 0, JOB_BATCH = 1 };

struct Job {
  uint64_t ticket;
  uint32_t kind;
  std::vector<uint64_t> words;
  uint64_t arrive_ns = 0;  // batch: moved off the ring (this thread)
};

struct Worker {
  Api api{};
  CUdevice device = 0;
  CUcontext ctx = nullptr;
  CUstream stream = nullptr;
  Job* ring[kRingSlots] = {};
  std::atomic<uint64_t> head{0};  // next ticket to enqueue (producer)
  std::atomic<uint64_t> tail{0};  // jobs consumed (consumer)
  std::atomic<uint64_t> recorded{0};  // last ticket whose event record was issued
  std::atomic<uint64_t> failed{0};    // first failed ticket (0: none)
  std::atomic<int> error{0};
  std::atomic<bool> sleeping{false};
  std::atomic<bool> stop{false};
  // Set once by dt_publish_worker_set_probe; null unless profiling.
  std::atomic<dt_probe::Probe*> probe{nullptr};
  std::mutex mutex;
  std::condition_variable wake;
  std::thread thread;

  // D8 send/completion loop; null callbacks until enable_batches.
  void (*wake_set)(void*, uint32_t) = nullptr;
  void* wake_word = nullptr;
  uint32_t done_bit = 0;
  int (*landing_release)(void*, uint32_t, uint32_t, uint64_t) = nullptr;
  void* ingress = nullptr;
  uint64_t done[kDoneSlots * kDoneWords] = {};
  std::atomic<uint64_t> done_head{0};  // records written (this thread)
  std::atomic<uint64_t> done_tail{0};  // records drained (scheduler)
  std::atomic<uint64_t> done_seq{0};   // last batch seq whose GPU work is done
  std::atomic<uint64_t> outstanding{0};  // descriptors not yet completed
  // Wall time this thread spent with a descriptor outstanding, and the
  // number of passes over it: the send loop's added CPU cost (cumulative).
  std::atomic<uint64_t> poll_ns{0};
  std::atomic<uint64_t> polls{0};

  // D9 pushes; set by enable_batches.
  std::atomic<bool> batches_on{false};
  CUstream late = nullptr;
  uint64_t push_const[K_PUSH_CONSTS] = {};
  uint64_t push_done[kPushDoneSlots * kPushDoneWords] = {};
  std::atomic<uint64_t> push_done_head{0};  // this thread
  std::atomic<uint64_t> push_done_tail{0};  // scheduler
  std::atomic<bool> expire_held{false};     // shutdown: give up held rows now
  // Held-list instrumentation: rows held now, the most at once, rows ever
  // held, and how the held ones ended; push work not yet PUSH_DONE.
  std::atomic<uint64_t> held_now{0}, held_max{0}, held_total{0};
  std::atomic<uint64_t> held_pushed{0}, held_cancelled{0}, held_expired{0};
  std::atomic<uint64_t> late_rows{0}, cross_rows{0};
  std::atomic<uint64_t> pushes_outstanding{0};
};

// The thread-local push state of serve() (no other thread reads it).
struct TrackedJob {
  uint64_t ticket;
  CUevent event;  // the scheduler's, alive until its PUSH_DONE is absorbed
  uint64_t issued_ns;
};
struct LateJob {
  CUevent event;  // pooled, owned by this thread
  uint64_t post_fn, post_arg, seq, issued_ns;
  std::vector<uint32_t> rows;
};
struct HeldRow {
  uint64_t seq;
  uint32_t row;
  uint64_t words[kPushRowWords];
};
struct PushState {
  std::deque<TrackedJob> jobs;  // FIFO on `stream`
  std::deque<LateJob> late;     // FIFO on `late`
  std::vector<HeldRow> held;
  std::vector<CUevent> pool;
};

CUresult run_job(Worker* w, const Job& job, dt_probe::Probe* p) {
  const uint64_t* q = job.words.data();
  CUevent event = reinterpret_cast<CUevent>(q[0]);
  const uint64_t copies = q[1], words = q[2];
  CUresult rc = timed(p, dt_probe::STREAM_WAIT_EVENT,
                      [&] { return w->api.cuStreamWaitEvent(w->stream, event, 0); });
  const uint64_t* c = q + kHeaderWords;
  for (uint64_t i = 0; rc == 0 && i < copies; ++i, c += 3) {
    rc = timed(p, dt_probe::MEMCPY_ASYNC,
               [&] { return w->api.cuMemcpyAsync(c[0], c[1], c[2], w->stream); });
  }
  for (uint64_t i = 0; rc == 0 && i < words; ++i, c += 2) {
    rc = timed(p, dt_probe::WRITE_VALUE32, [&] {
      return w->api.cuStreamWriteValue32(w->stream, c[0], (uint32_t)c[1], 0);
    });
  }
  if (rc == 0 && q[3]) {
    rc = timed(p, dt_probe::LAUNCH_HOST_FUNC, [&] {
      return w->api.cuLaunchHostFunc(w->stream, reinterpret_cast<CUhostFn>(q[3]),
                                     reinterpret_cast<void*>(q[4]));
    });
  }
  if (rc == 0)
    rc = timed(p, dt_probe::EVENT_RECORD, [&] { return w->api.cuEventRecord(event, w->stream); });
  return rc;
}

// One DONE record (design 7.2 step 4).  The scheduler is its only reader; a
// full queue is an invariant violation, not a stall, so it breaks the worker.
// False once the queue is full, which is an invariant violation (the
// scheduler drains it on every wake, and at most two batches plus its own
// fences can be outstanding), so the caller breaks the worker.
bool emit_done(Worker* w, uint64_t seq, uint64_t gpu_done_ns, uint64_t materialize_ns,
               uint64_t proposal_ns, uint64_t error) {
  const uint64_t head = w->done_head.load(std::memory_order_relaxed);
  if (head - w->done_tail.load(std::memory_order_acquire) >= kDoneSlots) return false;
  uint64_t* r = w->done + (head % kDoneSlots) * kDoneWords;
  r[0] = seq;
  r[1] = gpu_done_ns;
  r[2] = materialize_ns;
  r[3] = proposal_ns;
  r[4] = error;
  w->done_head.store(head + 1, std::memory_order_release);
  return true;
}

// Port of DraftCandidatePushTransport._publication_words (dt_candidate_p2p.py)
// for one row with a completion record: (address, value) pairs in issue order,
// the metadata generation last of the five metadata words and the record's
// seq word (high half, then low) last of all.  Byte-identical to the Python
// generator by test (test_dt_send_loop_push.py).
void publication_words(uint64_t* out, uint64_t metadata, uint64_t record, uint64_t nonce,
                       uint64_t generation, uint64_t valid_length, uint64_t root_token,
                       uint64_t transfer_hash, uint64_t producer_tag) {
  const uint64_t lo = nonce & 0xFFFFFFFFu, hi = (nonce >> 32) & 0xFFFFFFFFu;
  const uint64_t token = root_token & 0xFFFFFFFFu;
  const uint64_t pairs[kPublicationWords][2] = {
      // NONCE_LOW, NONCE_HIGH, VALID_LENGTH, ROOT_TOKEN, then GENERATION (0).
      {metadata + 4, lo}, {metadata + 8, hi}, {metadata + 12, valid_length},
      {metadata + 16, token}, {metadata, generation},
      {record + 8, generation}, {record + 12, valid_length},
      {record + 16, transfer_hash & 0xFFFFFFFFu}, {record + 20, transfer_hash >> 32},
      {record + 24, token}, {record + 28, producer_tag},
      {record + 32, 0}, {record + 36, 0}, {record + 40, 0}, {record + 44, 0},
      {record + 4, hi}, {record, lo},
  };
  memcpy(out, pairs, sizeof(pairs));
}

// Port of DraftRingEndpoint.destination (dt_direct_ring.py): the seqlock read
// of one request's destination from its record's region entry.  False: the
// validity word is not this seq (not written yet, or lapped).
struct Dest {
  uint32_t slot, generation;
  uint64_t nonce, data_offset, metadata_offset;
};
bool read_region(const uint64_t* k, const uint8_t* entry, uint64_t seq, uint64_t position,
                 Dest* d) {
  const volatile uint64_t* word =
      reinterpret_cast<const volatile uint64_t*>(entry + k[K_DEST_SEQ_AT]);
  for (int attempt = 0; attempt < 2; ++attempt) {
    const uint64_t first = *word;
    if (first != seq) return false;
    std::atomic_thread_fence(std::memory_order_acquire);
    const uint8_t* triple = entry + position * k[K_DEST_ENTRY_BYTES];
    memcpy(&d->slot, triple, 4);
    memcpy(&d->generation, triple + 4, 4);
    memcpy(&d->nonce, triple + 8, 8);
    memcpy(&d->data_offset, entry + k[K_DEST_DATA_AT], 8);
    memcpy(&d->metadata_offset, entry + k[K_DEST_META_AT], 8);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (*word == first) return true;
  }
  return false;
}

// Breaks the worker for good (design 7.5): no job or push is issued after.
void fail(Worker* w, int rc) {
  int expected = 0;
  w->error.compare_exchange_strong(expected, rc);
  uint64_t none = 0;
  w->failed.compare_exchange_strong(none, 1);
}

void emit_push_done(Worker* w, uint64_t kind, uint64_t id, uint64_t row, uint64_t outcome,
                    uint64_t issued_ns) {
  const uint64_t head = w->push_done_head.load(std::memory_order_relaxed);
  if (head - w->push_done_tail.load(std::memory_order_acquire) >= kPushDoneSlots) {
    fail(w, -4);  // the scheduler drains on every wake: an invariant violation
    return;
  }
  uint64_t* r = w->push_done + (head % kPushDoneSlots) * kPushDoneWords;
  r[0] = kind;
  r[1] = id;
  r[2] = row;
  r[3] = outcome;
  r[4] = issued_ns;
  r[5] = dt_probe::now_ns();
  w->push_done_head.store(head + 1, std::memory_order_release);
  w->pushes_outstanding.fetch_sub(1, std::memory_order_relaxed);
}

// The copies and words of one late job, collected while resolving rows.
struct LateWork {
  std::vector<uint64_t> copies;  // (destination, source, bytes)
  std::vector<uint64_t> words;   // (address, value)
  std::vector<uint32_t> rows;
};

// Issue `work` on the late stream and track it; on a broken worker (or a
// failed driver call) its rows are reported FAILED instead.
void issue_late(Worker* w, PushState& s, dt_probe::Probe* p, uint64_t seq, LateWork& work,
                uint64_t post_fn, uint64_t post_arg) {
  if (work.rows.empty()) return;
  CUresult rc = w->failed.load() ? -1 : 0;
  CUevent event = nullptr;
  if (rc == 0 && !s.pool.empty()) {
    event = s.pool.back();
    s.pool.pop_back();
  } else if (rc == 0) {
    rc = w->api.cuEventCreate(&event, 0x2);  // CU_EVENT_DISABLE_TIMING
  }
  for (size_t i = 0; rc == 0 && i < work.copies.size(); i += 3) {
    rc = timed(p, dt_probe::MEMCPY_ASYNC, [&] {
      return w->api.cuMemcpyAsync(work.copies[i], work.copies[i + 1], work.copies[i + 2],
                                  w->late);
    });
  }
  for (size_t i = 0; rc == 0 && i < work.words.size(); i += 2) {
    rc = timed(p, dt_probe::WRITE_VALUE32, [&] {
      return w->api.cuStreamWriteValue32(w->late, work.words[i], (uint32_t)work.words[i + 1], 0);
    });
  }
  if (rc == 0)
    rc = timed(p, dt_probe::EVENT_RECORD, [&] { return w->api.cuEventRecord(event, w->late); });
  if (rc != 0) {
    fail(w, rc);
    if (event) s.pool.push_back(event);
    for (uint32_t row : work.rows) emit_push_done(w, PUSH_DONE_ROW, seq, row, FAILED, 0);
    return;
  }
  s.late.push_back(LateJob{event, post_fn, post_arg, seq, dt_probe::now_ns(), work.rows});
}

// One LATE_LOCAL row: true once it is settled (its copy and words are in
// `work`, or its PUSH_DONE is emitted), false while its destination is absent.
bool resolve_late(Worker* w, uint64_t seq, uint32_t row, const uint64_t* q, bool held,
                  LateWork& work) {
  const uint64_t* k = w->push_const;
  Dest d{};
  if (!read_region(k, reinterpret_cast<const uint8_t*>(q[3]), q[4], q[5], &d)) {
    if (!w->expire_held.load(std::memory_order_relaxed) && dt_probe::now_ns() < q[14])
      return false;
    if (held) w->held_expired.fetch_add(1, std::memory_order_relaxed);
    emit_push_done(w, PUSH_DONE_ROW, seq, row, EXPIRED, 0);
    return true;
  }
  if (d.slot == 0 && d.generation == 0 && d.nonce == 0) {
    // The Target's tombstone: the row ended in its wave (design 7.3).
    if (held) w->held_cancelled.fetch_add(1, std::memory_order_relaxed);
    emit_push_done(w, PUSH_DONE_ROW, seq, row, CANCELLED, 0);
    return true;
  }
  // _ring_destination_ref + _mapping + _validate_destination: an invalid
  // destination is an invariant violation, so it fails stop.
  const uint64_t bytes = q[2];
  const uint64_t data_at = d.data_offset + (uint64_t)d.slot * bytes;
  const uint64_t metadata_at = d.metadata_offset + (uint64_t)d.slot * k[K_METADATA_WORDS] * 4;
  if (d.generation == 0 || d.generation > 0x7FFFFFFFu || d.nonce == 0 ||
      d.nonce >= (1ull << 63) || data_at + bytes > q[7] ||
      metadata_at + k[K_METADATA_WORDS] * 4 > q[9] || d.slot >= q[11]) {
    fail(w, -3);
    emit_push_done(w, PUSH_DONE_ROW, seq, row, FAILED, 0);
    return true;
  }
  if (held) w->held_pushed.fetch_add(1, std::memory_order_relaxed);
  work.copies.insert(work.copies.end(), {q[6] + data_at, q[1], bytes});
  const size_t at = work.words.size();
  work.words.resize(at + 2 * kPublicationWords);
  publication_words(work.words.data() + at, q[8] + metadata_at,
                    q[10] + (uint64_t)d.slot * k[K_RECORD_BYTES], d.nonce, d.generation,
                    bytes / 4, q[13], q[12], k[K_PRODUCER_TAG]);
  work.rows.push_back(row);
  return true;
}

// Design 7.2 step 3 for one completed descriptor: LATE_LOCAL rows resolved
// now or held, CROSS_HOST rows copied into their mirror and posted once that
// copy has run.
void issue_batch_pushes(Worker* w, PushState& s, const Job& job, dt_probe::Probe* p) {
  const uint64_t* q = job.words.data();
  const uint64_t* push = q + kBatchHeaderWords + 3 * q[5];
  const uint64_t rows = push[0];
  if (!rows) return;
  const uint64_t seq = q[0];
  LateWork work;
  const uint64_t* r = push + kPushHeaderWords;
  for (uint32_t i = 0; i < rows; ++i, r += kPushRowWords) {
    if (r[0] == PUSH_CROSS_HOST) {
      work.copies.insert(work.copies.end(), {r[3], r[1], r[2]});
      work.rows.push_back(i);
      w->cross_rows.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    w->late_rows.fetch_add(1, std::memory_order_relaxed);
    if (!resolve_late(w, seq, i, r, false, work)) {
      HeldRow h{seq, i, {}};
      memcpy(h.words, r, sizeof(h.words));
      s.held.push_back(h);
      w->held_total.fetch_add(1, std::memory_order_relaxed);
    }
  }
  const uint64_t held = s.held.size();
  w->held_now.store(held, std::memory_order_relaxed);
  if (held > w->held_max.load(std::memory_order_relaxed))
    w->held_max.store(held, std::memory_order_relaxed);
  issue_late(w, s, p, seq, work, push[1], push[2]);
}

// One pass over the outstanding push work: tracked jobs and late jobs in
// FIFO order (each stream completes in order), then the held rows.  True if
// any PUSH_DONE record was emitted (the caller wakes the scheduler).
bool service_pushes(Worker* w, PushState& s, dt_probe::Probe* p) {
  const uint64_t before = w->push_done_head.load(std::memory_order_relaxed);
  while (!s.jobs.empty()) {
    const TrackedJob& job = s.jobs.front();
    const CUresult rc = timed(p, dt_probe::EVENT_QUERY,
                              [&] { return w->api.cuEventQuery(job.event); });
    if (rc == kNotReady) break;
    if (rc != 0) fail(w, rc);
    emit_push_done(w, PUSH_DONE_JOB, job.ticket, 0, rc ? FAILED : PUSHED, job.issued_ns);
    s.jobs.pop_front();
  }
  while (!s.late.empty()) {
    LateJob& job = s.late.front();
    const CUresult rc = timed(p, dt_probe::EVENT_QUERY,
                              [&] { return w->api.cuEventQuery(job.event); });
    if (rc == kNotReady) break;
    if (rc != 0) fail(w, rc);
    // The mirror rows are written: post the cross-host WRITEs from here
    // (dt_rdma_post_candidates; it writes each group's status itself).
    if (rc == 0 && job.post_fn) reinterpret_cast<CUhostFn>(job.post_fn)((void*)job.post_arg);
    for (uint32_t row : job.rows)
      emit_push_done(w, PUSH_DONE_ROW, job.seq, row, rc ? FAILED : PUSHED, job.issued_ns);
    s.pool.push_back(job.event);
    s.late.pop_front();
  }
  if (!s.held.empty()) {
    // One late job per (pass, batch) of rows that resolved.
    size_t kept = 0;
    for (size_t i = 0; i < s.held.size();) {
      const uint64_t seq = s.held[i].seq;
      LateWork work;
      for (; i < s.held.size() && s.held[i].seq == seq; ++i) {
        if (!resolve_late(w, seq, s.held[i].row, s.held[i].words, true, work))
          s.held[kept++] = s.held[i];
      }
      issue_late(w, s, p, seq, work, 0, 0);
    }
    s.held.resize(kept);
    w->held_now.store(kept, std::memory_order_relaxed);
  }
  return w->push_done_head.load(std::memory_order_relaxed) != before;
}

// A batch descriptor whose event completed (design 7.2).  `gpu_done_ns` was
// taken the moment cuEventQuery reported it, before any driver call here.
void complete_batch(Worker* w, const Job& job, uint64_t gpu_done_ns, dt_probe::Probe* p) {
  const uint64_t* q = job.words.data();
  const uint64_t seq = q[0];
  uint64_t materialize_ns = 0, proposal_ns = 0, error = 0;
  if (seq && q[2] && q[3] && q[4]) {
    float ms = 0.0f;
    CUresult rc = timed(p, dt_probe::EVENT_ELAPSED, [&] {
      return w->api.cuEventElapsedTime(&ms, (CUevent)q[2], (CUevent)q[3]);
    });
    if (rc == 0) {
      materialize_ns = (uint64_t)((double)ms * 1e6);
      rc = timed(p, dt_probe::EVENT_ELAPSED, [&] {
        return w->api.cuEventElapsedTime(&ms, (CUevent)q[3], (CUevent)q[4]);
      });
      if (rc == 0) proposal_ns = (uint64_t)((double)ms * 1e6);
    }
    if (rc != 0) error = (uint64_t)rc;
  }
  if (!emit_done(w, seq, gpu_done_ns, materialize_ns, proposal_ns, error)) {
    // No record: done_seq must not move past a batch the scheduler cannot
    // see, and its landing releases stay unsent (the buffer is retained).
    w->error.store(-1);
    w->failed.store(job.ticket);
    return;
  }
  if (seq) w->done_seq.store(seq, std::memory_order_release);
  if (w->wake_set) {
    timed(p, dt_probe::WAKE_SET, [&] {
      w->wake_set(w->wake_word, w->done_bit);
      return 0;
    });
  }
  // The landing placements this batch read are free now that its GPU work is
  // done (design 7.2 step 2): ingress owns the occupancy and applies them.
  const uint64_t releases = q[5];
  const uint64_t* rel = q + kBatchHeaderWords;
  for (uint64_t i = 0; i < releases; ++i, rel += 3) {
    if (w->landing_release == nullptr ||
        w->landing_release(w->ingress, (uint32_t)rel[0], (uint32_t)rel[1], rel[2]) != 0) {
      w->error.store(-2);
      w->failed.store(job.ticket);
      return;
    }
  }
}

void serve(Worker* w) {
  w->api.cuCtxSetCurrent(w->ctx);
  // Descriptors outstanding, in submit order; owned by this thread alone.
  std::deque<Job*> batches;
  PushState s;
  uint64_t poll_started_ns = 0;
  // Bounded wait: GPU done of the last batch, and per class (0: nothing was
  // ahead at its arrival, 1: queued behind another) the last spans from its
  // start (arrival or the previous GPU done) to its GPU done.
  uint64_t last_done_ns = 0;
  uint64_t spans[2][kWindow] = {};
  int spans_seen[2] = {0, 0};
  while (true) {
    dt_probe::Probe* p = w->probe.load(std::memory_order_acquire);
    bool progress = false;
    uint64_t tail = w->tail.load(std::memory_order_relaxed);
    while (tail != w->head.load(std::memory_order_acquire)) {
      Job* job = w->ring[tail % kRingSlots];
      if (job->kind == JOB_BATCH) {
        if (batches.empty() && !last_done_ns && !spans_seen[0])
          prctl(PR_SET_TIMERSLACK, 1000);  // the bounded wait's sleeps are 50 us
        job->arrive_ns = dt_probe::now_ns();
        const uint64_t* q = job->words.data();
        w->pushes_outstanding.fetch_add(q[kBatchHeaderWords + 3 * q[5]],
                                        std::memory_order_relaxed);
        batches.push_back(job);
      } else {
        // D9: every job queued under the send loop ends in one PUSH_DONE
        // record, a failed or skipped one (broken worker) at once.
        const bool tracked = w->batches_on.load(std::memory_order_acquire);
        if (tracked) w->pushes_outstanding.fetch_add(1, std::memory_order_relaxed);
        bool issued = false;
        if (w->failed.load() == 0) {
          // arg: the copy count (candidate rows) of the job.
          dt_probe::stamp(p, dt_probe::JOB_TAKEN, -1, job->ticket, job->words[1]);
          const CUresult rc = run_job(w, *job, p);
          if (rc != 0) {
            w->error.store(rc);
            w->failed.store(job->ticket);
          } else {
            dt_probe::stamp(p, dt_probe::JOB_ISSUED, -1, job->ticket, job->words[1]);
            w->recorded.store(job->ticket, std::memory_order_release);
            issued = true;
          }
        }
        if (tracked && issued) {
          s.jobs.push_back(TrackedJob{job->ticket, (CUevent)job->words[0], dt_probe::now_ns()});
        } else if (tracked) {
          emit_push_done(w, PUSH_DONE_JOB, job->ticket, 0, FAILED, 0);
          if (w->wake_set) w->wake_set(w->wake_word, w->done_bit);
        }
        delete job;
      }
      w->tail.store(++tail, std::memory_order_release);
      progress = true;
    }
    if (!batches.empty()) {
      // Wall time this thread spends with a descriptor outstanding, and the
      // number of passes it takes: accumulated per pass, so a thread that
      // never runs dry (steady state at depth 2) still reports its cost.
      const uint64_t pass_started_ns = dt_probe::now_ns();
      if (poll_started_ns) {
        w->poll_ns.fetch_add(pass_started_ns - poll_started_ns, std::memory_order_relaxed);
        w->polls.fetch_add(1, std::memory_order_relaxed);
      }
      poll_started_ns = pass_started_ns;
      Job* head = batches.front();
      const CUresult rc = timed(p, dt_probe::EVENT_QUERY, [&] {
        return w->api.cuEventQuery((CUevent)head->words[1]);
      });
      if (rc == 0) {
        const uint64_t gpu_done_ns = dt_probe::now_ns();
        dt_probe::stamp(p, dt_probe::BATCH_DONE, -1, head->words[0], batches.size());
        complete_batch(w, *head, gpu_done_ns, p);
        issue_batch_pushes(w, s, *head, p);
        if (head->words[0]) {
          const int c = head->arrive_ns >= last_done_ns ? 0 : 1;
          const uint64_t start = c ? last_done_ns : head->arrive_ns;
          spans[c][spans_seen[c]++ % kWindow] = gpu_done_ns - start;
          last_done_ns = gpu_done_ns;
        }
        delete head;
        batches.pop_front();
        w->outstanding.fetch_sub(1, std::memory_order_release);
        progress = true;
      } else if (rc != kNotReady) {
        // A failed query cannot prove the batch's GPU writes have stopped:
        // report it and stop watching (design 7.5; the scheduler quarantines).
        w->error.store(rc);
        w->failed.store(head->ticket);
        if (emit_done(w, head->words[0], dt_probe::now_ns(), 0, 0, (uint64_t)rc) &&
            w->wake_set)
          w->wake_set(w->wake_word, w->done_bit);
        issue_batch_pushes(w, s, *head, p);  // broken: its rows report FAILED
        delete head;
        batches.pop_front();
        w->outstanding.fetch_sub(1, std::memory_order_release);
        progress = true;
      }
      if (batches.empty()) {
        w->poll_ns.fetch_add(dt_probe::now_ns() - poll_started_ns, std::memory_order_relaxed);
        w->polls.fetch_add(1, std::memory_order_relaxed);
        poll_started_ns = 0;
      }
    }
    if (service_pushes(w, s, p)) {
      if (w->wake_set) w->wake_set(w->wake_word, w->done_bit);
      progress = true;
    }
    if (progress) continue;
    // destroy() drained first, so stopping with descriptors left means the
    // worker is broken; drop them rather than wait out their events.
    if (w->stop.load()) {
      for (Job* job : batches) delete job;
      for (CUevent event : s.pool) w->api.cuEventDestroy(event);
      for (LateJob& job : s.late) w->api.cuEventDestroy(job.event);
      return;
    }
    if (!batches.empty() || !s.jobs.empty() || !s.late.empty() || !s.held.empty()) {
      // Spin while a late job's tiny copy runs (its cross-host post waits on
      // it) or the head batch is inside its predicted completion window, or
      // with no prediction yet; otherwise sleep until the window opens, at
      // most kPollSleep.  Spinning throughout cost ~32% of a core (#277).
      bool spin = !s.late.empty();
      auto sleep = std::chrono::nanoseconds(kPollSleep);
      if (!spin && !batches.empty()) {
        const Job* head = batches.front();
        const int c = head->arrive_ns >= last_done_ns ? 0 : 1;
        const uint64_t start = c ? last_done_ns : head->arrive_ns;
        const int seen = spans_seen[c] < kWindow ? spans_seen[c] : kWindow;
        uint64_t shortest = UINT64_MAX;
        for (int i = 0; i < seen; ++i) shortest = spans[c][i] < shortest ? spans[c][i] : shortest;
        const uint64_t now = dt_probe::now_ns();
        if (!seen || start + shortest <= now + kGuardNs) {
          spin = true;
        } else if (start + shortest - kGuardNs - now < (uint64_t)sleep.count()) {
          sleep = std::chrono::nanoseconds(start + shortest - kGuardNs - now);
        }
      }
      if (spin) {
        sched_yield();
      } else {
        std::unique_lock<std::mutex> lock(w->mutex);
        w->sleeping.store(true);
        w->wake.wait_for(lock, sleep, [&] { return w->stop.load() || tail != w->head.load(); });
        w->sleeping.store(false);
      }
      continue;
    }
    const auto spin_end = std::chrono::steady_clock::now() + kSpin;
    bool more = false;
    while (std::chrono::steady_clock::now() < spin_end) {
      if (tail != w->head.load(std::memory_order_acquire)) {
        more = true;
        break;
      }
      sched_yield();
    }
    if (!more) {
      std::unique_lock<std::mutex> lock(w->mutex);
      w->sleeping.store(true);
      w->wake.wait_for(lock, std::chrono::milliseconds(1), [&] {
        return w->stop.load() || tail != w->head.load();
      });
      w->sleeping.store(false);
    }
  }
}

}  // namespace

extern "C" {

void* dt_publish_worker_create(int device, int priority) {
  Worker* w = new Worker();
  CUdevice dev;
  if (!load_api(&w->api) || w->api.cuInit(0) != 0 || w->api.cuDeviceGet(&dev, device) != 0 ||
      w->api.cuDevicePrimaryCtxRetain(&w->ctx, dev) != 0) {
    delete w;
    return nullptr;
  }
  w->device = dev;
  // CU_STREAM_NON_BLOCKING, like torch's streams.
  if (w->api.cuCtxSetCurrent(w->ctx) != 0 ||
      w->api.cuStreamCreateWithPriority(&w->stream, 0x1, priority) != 0) {
    w->api.cuDevicePrimaryCtxRelease(dev);
    delete w;
    return nullptr;
  }
  w->thread = std::thread(serve, w);
  return w;
}

// Attach a dt_probe_create probe (owned by the caller, outliving the
// worker).  Once only: the thread may already be reading it.
int dt_publish_worker_set_probe(void* handle, void* probe) {
  Worker* w = static_cast<Worker*>(handle);
  dt_probe::Probe* expected = nullptr;
  return w->probe.compare_exchange_strong(expected, static_cast<dt_probe::Probe*>(probe),
                                          std::memory_order_release)
             ? 0
             : -1;
}

namespace {

// Shared producer side of the ring; the caller has checked `count`.
uint64_t enqueue(Worker* w, uint32_t kind, const uint64_t* words, uint64_t count) {
  if (w->failed.load() != 0) return 0;
  const uint64_t head = w->head.load(std::memory_order_relaxed);
  while (head - w->tail.load(std::memory_order_acquire) >= kRingSlots) {
    if (w->failed.load() != 0) return 0;
    sched_yield();
  }
  const uint64_t ticket = head + 1;
  w->ring[head % kRingSlots] = new Job{ticket, kind, std::vector<uint64_t>(words, words + count)};
  // Published: the consumer owns the job from here and may already have
  // deleted it, so the ticket must come from this thread's own copy.
  w->head.store(head + 1, std::memory_order_seq_cst);
  if (w->sleeping.load(std::memory_order_seq_cst)) {
    std::lock_guard<std::mutex> lock(w->mutex);
    w->wake.notify_one();
  }
  return ticket;
}

}  // namespace

// Queue one job; returns its ticket (>= 1), or 0 once the worker is broken.
// Blocks while the ring is full.
uint64_t dt_publish_worker_submit(void* handle, const uint64_t* words, uint64_t count) {
  Worker* w = static_cast<Worker*>(handle);
  if (count < kHeaderWords) return 0;
  return enqueue(w, JOB_PUSH, words, count);
}

// D8: turn the send/completion loop on.  `wake_set(wake, done_bit)` wakes the
// scheduler at a GPU done, and `landing_release(ingress, k, gen, count)` is
// the ingress worker's release queue.  D9: `push_const` holds the
// K_PUSH_CONSTS layout constants of the push rows, and the late stream is
// created here.  Once only, before any descriptor.
int dt_publish_worker_enable_batches(void* handle, void (*wake_set)(void*, uint32_t),
                                     void* wake, uint32_t done_bit,
                                     int (*landing_release)(void*, uint32_t, uint32_t,
                                                            uint64_t),
                                     void* ingress, const uint64_t* push_const,
                                     int priority) {
  Worker* w = static_cast<Worker*>(handle);
  if (w->landing_release != nullptr || landing_release == nullptr) return -1;
  if (w->api.cuCtxSetCurrent(w->ctx) != 0 ||
      w->api.cuStreamCreateWithPriority(&w->late, 0x1, priority) != 0)
    return -1;
  memcpy(w->push_const, push_const, sizeof(w->push_const));
  w->wake_set = wake_set;
  w->wake_word = wake;
  w->done_bit = done_bit;
  w->ingress = ingress;
  // Published last: the worker thread reads the callbacks only through a
  // descriptor, which cannot be on the ring before this call returns.
  std::atomic_thread_fence(std::memory_order_release);
  w->landing_release = landing_release;
  // Jobs queued from here on are tracked to a PUSH_DONE record.
  w->batches_on.store(true, std::memory_order_release);
  return 0;
}

// Queue one batch descriptor (see kBatchHeaderWords); returns its ticket, or
// 0 once the worker is broken.
uint64_t dt_publish_worker_submit_batch(void* handle, const uint64_t* words,
                                        uint64_t count) {
  Worker* w = static_cast<Worker*>(handle);
  if (count < kBatchHeaderWords + kPushHeaderWords) return 0;
  const uint64_t push_at = kBatchHeaderWords + 3 * words[5];
  if (count < push_at + kPushHeaderWords ||
      count != push_at + kPushHeaderWords + kPushRowWords * words[push_at])
    return 0;
  w->outstanding.fetch_add(1, std::memory_order_release);
  const uint64_t ticket = enqueue(w, JOB_BATCH, words, count);
  if (!ticket) w->outstanding.fetch_sub(1, std::memory_order_release);
  return ticket;
}

// Move up to `max` DONE records (kDoneWords u64 each) into `out`; the count.
uint64_t dt_publish_worker_done_drain(void* handle, uint64_t* out, uint64_t max) {
  Worker* w = static_cast<Worker*>(handle);
  const uint64_t tail = w->done_tail.load(std::memory_order_relaxed);
  const uint64_t head = w->done_head.load(std::memory_order_acquire);
  const uint64_t n = head - tail < max ? head - tail : max;
  for (uint64_t i = 0; i < n; ++i)
    __builtin_memcpy(out + kDoneWords * i, w->done + ((tail + i) % kDoneSlots) * kDoneWords,
                     kDoneWords * sizeof(uint64_t));
  w->done_tail.store(tail + n, std::memory_order_release);
  return n;
}

// Move up to `max` PUSH_DONE records (kPushDoneWords u64 each) into `out`.
uint64_t dt_publish_worker_push_done_drain(void* handle, uint64_t* out, uint64_t max) {
  Worker* w = static_cast<Worker*>(handle);
  const uint64_t tail = w->push_done_tail.load(std::memory_order_relaxed);
  const uint64_t head = w->push_done_head.load(std::memory_order_acquire);
  const uint64_t n = head - tail < max ? head - tail : max;
  for (uint64_t i = 0; i < n; ++i)
    __builtin_memcpy(out + kPushDoneWords * i,
                     w->push_done + ((tail + i) % kPushDoneSlots) * kPushDoneWords,
                     kPushDoneWords * sizeof(uint64_t));
  w->push_done_tail.store(tail + n, std::memory_order_release);
  return n;
}

// Shutdown: every held row (and every later one) expires at its next read.
void dt_publish_worker_expire_held(void* handle) {
  static_cast<Worker*>(handle)->expire_held.store(true);
}

// {done_seq, descriptors outstanding, poll waits, cumulative poll ns, thread
//  CPU ns, rows held now, most held at once, rows ever held, held rows that
//  were pushed / cancelled / expired, LATE_LOCAL rows, CROSS_HOST rows, push
//  work not yet PUSH_DONE}.
void dt_publish_worker_batch_stats(void* handle, uint64_t* out) {
  Worker* w = static_cast<Worker*>(handle);
  out[0] = w->done_seq.load(std::memory_order_acquire);
  out[1] = w->outstanding.load(std::memory_order_acquire);
  out[2] = w->polls.load(std::memory_order_relaxed);
  out[3] = w->poll_ns.load(std::memory_order_relaxed);
  clockid_t clock;
  timespec ts{};
  if (pthread_getcpuclockid(w->thread.native_handle(), &clock) == 0)
    clock_gettime(clock, &ts);
  out[4] = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
  const std::atomic<uint64_t>* held[] = {
      &w->held_now, &w->held_max, &w->held_total, &w->held_pushed, &w->held_cancelled,
      &w->held_expired, &w->late_rows, &w->cross_rows, &w->pushes_outstanding};
  for (size_t i = 0; i < sizeof(held) / sizeof(held[0]); ++i)
    out[5 + i] = held[i]->load(std::memory_order_relaxed);
}

// Test hook (no CUDA): one row's publication words, as the send loop issues
// them; returns their pair count.
uint64_t dt_publish_words(uint64_t* out, uint64_t metadata, uint64_t record, uint64_t nonce,
                          uint64_t generation, uint64_t valid_length, uint64_t root_token,
                          uint64_t transfer_hash, uint64_t producer_tag) {
  publication_words(out, metadata, record, nonce, generation, valid_length, root_token,
                    transfer_hash, producer_tag);
  return kPublicationWords;
}

// 1: the ticket's completion record is issued; 0: not yet; < 0: the job
// failed with driver error -status (or a later job's ticket after a failure).
int dt_publish_worker_status(void* handle, uint64_t ticket) {
  Worker* w = static_cast<Worker*>(handle);
  if (w->recorded.load(std::memory_order_acquire) >= ticket) return 1;
  const uint64_t failed = w->failed.load();
  if (failed != 0 && ticket >= failed) {
    const int error = w->error.load();
    return error > 0 ? -error : -1;
  }
  return 0;
}

// Wait until every queued job was issued and every batch descriptor
// completed (its landing releases are then with ingress), then drain the
// stream.  Returns 0 or the first driver error.  A latched error stops the
// wait: the worker issues nothing more.
int dt_publish_worker_drain(void* handle) {
  Worker* w = static_cast<Worker*>(handle);
  const uint64_t head = w->head.load();
  while (w->tail.load(std::memory_order_acquire) < head) sched_yield();
  // A broken worker still completes descriptors (it only stops *issuing*),
  // so this terminates for the same reason the ring drain does.
  while (w->outstanding.load(std::memory_order_acquire) != 0) sched_yield();
  w->api.cuCtxSetCurrent(w->ctx);
  CUresult rc = w->api.cuStreamSynchronize(w->stream);
  if (rc == 0 && w->late) rc = w->api.cuStreamSynchronize(w->late);
  return w->error.load() ? w->error.load() : rc;
}

int dt_publish_worker_destroy(void* handle) {
  Worker* w = static_cast<Worker*>(handle);
  if (!w) return 0;
  const int rc = dt_publish_worker_drain(w);
  w->stop.store(true);
  {
    std::lock_guard<std::mutex> lock(w->mutex);
    w->wake.notify_one();
  }
  w->thread.join();
  w->api.cuStreamDestroy(w->stream);
  if (w->late) w->api.cuStreamDestroy(w->late);
  w->api.cuDevicePrimaryCtxRelease(w->device);
  delete w;
  return rc;
}

}  // extern "C"
