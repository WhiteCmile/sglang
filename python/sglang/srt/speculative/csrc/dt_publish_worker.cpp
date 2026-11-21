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

#include <dlfcn.h>
#include <sched.h>
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

enum JobKind : uint32_t { JOB_PUSH = 0, JOB_BATCH = 1 };

struct Job {
  uint64_t ticket;
  uint32_t kind;
  std::vector<uint64_t> words;
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
void emit_done(Worker* w, uint64_t seq, uint64_t gpu_done_ns, uint64_t materialize_ns,
               uint64_t proposal_ns, uint64_t error) {
  const uint64_t head = w->done_head.load(std::memory_order_relaxed);
  if (head - w->done_tail.load(std::memory_order_acquire) >= kDoneSlots) {
    w->error.store(-1);
    w->failed.store(head + 1);
    return;
  }
  uint64_t* r = w->done + (head % kDoneSlots) * kDoneWords;
  r[0] = seq;
  r[1] = gpu_done_ns;
  r[2] = materialize_ns;
  r[3] = proposal_ns;
  r[4] = error;
  w->done_head.store(head + 1, std::memory_order_release);
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
  emit_done(w, seq, gpu_done_ns, materialize_ns, proposal_ns, error);
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
  uint64_t poll_started_ns = 0;
  while (true) {
    dt_probe::Probe* p = w->probe.load(std::memory_order_acquire);
    bool progress = false;
    uint64_t tail = w->tail.load(std::memory_order_relaxed);
    while (tail != w->head.load(std::memory_order_acquire)) {
      Job* job = w->ring[tail % kRingSlots];
      if (job->kind == JOB_BATCH) {
        batches.push_back(job);
      } else {
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
          }
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
        delete head;
        batches.pop_front();
        w->outstanding.fetch_sub(1, std::memory_order_release);
        progress = true;
      } else if (rc != kNotReady) {
        // A failed query cannot prove the batch's GPU writes have stopped:
        // report it and stop watching (design 7.5; the scheduler quarantines).
        w->error.store(rc);
        w->failed.store(head->ticket);
        emit_done(w, head->words[0], dt_probe::now_ns(), 0, 0, (uint64_t)rc);
        if (w->wake_set) w->wake_set(w->wake_word, w->done_bit);
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
    if (progress) continue;
    // destroy() drained first, so stopping with descriptors left means the
    // worker is broken; drop them rather than wait out their events.
    if (w->stop.load()) {
      for (Job* job : batches) delete job;
      return;
    }
    if (!batches.empty()) {
      sched_yield();
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
  Job* job = new Job{head + 1, kind, std::vector<uint64_t>(words, words + count)};
  w->ring[head % kRingSlots] = job;
  w->head.store(head + 1, std::memory_order_seq_cst);
  if (w->sleeping.load(std::memory_order_seq_cst)) {
    std::lock_guard<std::mutex> lock(w->mutex);
    w->wake.notify_one();
  }
  return job->ticket;
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
// the ingress worker's release queue.  Once only, before any descriptor.
int dt_publish_worker_enable_batches(void* handle, void (*wake_set)(void*, uint32_t),
                                     void* wake, uint32_t done_bit,
                                     int (*landing_release)(void*, uint32_t, uint32_t,
                                                            uint64_t),
                                     void* ingress) {
  Worker* w = static_cast<Worker*>(handle);
  if (w->landing_release != nullptr || landing_release == nullptr) return -1;
  w->wake_set = wake_set;
  w->wake_word = wake;
  w->done_bit = done_bit;
  w->ingress = ingress;
  // Published last: the worker thread reads the callbacks only through a
  // descriptor, which cannot be on the ring before this call returns.
  std::atomic_thread_fence(std::memory_order_release);
  w->landing_release = landing_release;
  return 0;
}

// Queue one batch descriptor (see kBatchHeaderWords); returns its ticket, or
// 0 once the worker is broken.
uint64_t dt_publish_worker_submit_batch(void* handle, const uint64_t* words,
                                        uint64_t count) {
  Worker* w = static_cast<Worker*>(handle);
  if (count < kBatchHeaderWords || count != kBatchHeaderWords + 3 * words[5]) return 0;
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

// {done_seq, descriptors outstanding, poll waits, cumulative poll ns}.
void dt_publish_worker_batch_stats(void* handle, uint64_t* out) {
  Worker* w = static_cast<Worker*>(handle);
  out[0] = w->done_seq.load(std::memory_order_acquire);
  out[1] = w->outstanding.load(std::memory_order_acquire);
  out[2] = w->polls.load(std::memory_order_relaxed);
  out[3] = w->poll_ns.load(std::memory_order_relaxed);
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
  while (w->tail.load(std::memory_order_acquire) < head && w->failed.load() == 0)
    sched_yield();
  while (w->outstanding.load(std::memory_order_acquire) != 0 && w->failed.load() == 0)
    sched_yield();
  w->api.cuCtxSetCurrent(w->ctx);
  const CUresult rc = w->api.cuStreamSynchronize(w->stream);
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
  w->api.cuDevicePrimaryCtxRelease(w->device);
  delete w;
  return rc;
}

}  // extern "C"
