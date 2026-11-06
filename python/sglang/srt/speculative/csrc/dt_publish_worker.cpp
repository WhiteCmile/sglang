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

#include <dlfcn.h>
#include <sched.h>
#include <stdint.h>
#include <string.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace {

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
};

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

struct Job {
  uint64_t ticket;
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
  std::mutex mutex;
  std::condition_variable wake;
  std::thread thread;
};

CUresult run_job(Worker* w, const Job& job) {
  const uint64_t* q = job.words.data();
  CUevent event = reinterpret_cast<CUevent>(q[0]);
  const uint64_t copies = q[1], words = q[2];
  CUresult rc = w->api.cuStreamWaitEvent(w->stream, event, 0);
  const uint64_t* c = q + kHeaderWords;
  for (uint64_t i = 0; rc == 0 && i < copies; ++i, c += 3) {
    rc = w->api.cuMemcpyAsync(c[0], c[1], c[2], w->stream);
  }
  for (uint64_t i = 0; rc == 0 && i < words; ++i, c += 2) {
    rc = w->api.cuStreamWriteValue32(w->stream, c[0], (uint32_t)c[1], 0);
  }
  if (rc == 0 && q[3]) {
    rc = w->api.cuLaunchHostFunc(w->stream, reinterpret_cast<CUhostFn>(q[3]),
                                 reinterpret_cast<void*>(q[4]));
  }
  if (rc == 0) rc = w->api.cuEventRecord(event, w->stream);
  return rc;
}

void serve(Worker* w) {
  w->api.cuCtxSetCurrent(w->ctx);
  while (true) {
    const uint64_t tail = w->tail.load(std::memory_order_relaxed);
    if (tail == w->head.load(std::memory_order_acquire)) {
      if (w->stop.load()) return;
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
      continue;
    }
    Job* job = w->ring[tail % kRingSlots];
    if (w->failed.load() == 0) {
      const CUresult rc = run_job(w, *job);
      if (rc != 0) {
        w->error.store(rc);
        w->failed.store(job->ticket);
      } else {
        w->recorded.store(job->ticket, std::memory_order_release);
      }
    }
    delete job;
    w->tail.store(tail + 1, std::memory_order_release);
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

// Queue one job; returns its ticket (>= 1), or 0 once the worker is broken.
// Blocks while the ring is full.
uint64_t dt_publish_worker_submit(void* handle, const uint64_t* words, uint64_t count) {
  Worker* w = static_cast<Worker*>(handle);
  if (w->failed.load() != 0 || count < kHeaderWords) return 0;
  const uint64_t head = w->head.load(std::memory_order_relaxed);
  while (head - w->tail.load(std::memory_order_acquire) >= kRingSlots) {
    if (w->failed.load() != 0) return 0;
    sched_yield();
  }
  Job* job = new Job{head + 1, std::vector<uint64_t>(words, words + count)};
  w->ring[head % kRingSlots] = job;
  w->head.store(head + 1, std::memory_order_seq_cst);
  if (w->sleeping.load(std::memory_order_seq_cst)) {
    std::lock_guard<std::mutex> lock(w->mutex);
    w->wake.notify_one();
  }
  return job->ticket;
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

// Wait until every queued job was issued, then drain the stream.  Returns 0
// or the first driver error.
int dt_publish_worker_drain(void* handle) {
  Worker* w = static_cast<Worker*>(handle);
  const uint64_t head = w->head.load();
  while (w->tail.load(std::memory_order_acquire) < head) sched_yield();
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
