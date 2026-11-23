// DT Target cross-host slot-ready native sender (issue #313, redesign P5a).
//
// Replaces the Python `dt-rdma-ready-publisher` thread (dt_rdma.py
// _publisher_loop) behind DT_TARGET_READY_SENDER=native; the bytes on the
// wire are unchanged.  One native thread per TargetRdmaEndpoint:
//
//   1. pops one ready job {(slot, generation), producer CUevent, enqueued_ns}
//      off a lock-free SPSC ring (the scheduler thread is the only producer,
//      via dt_target_sender_submit);
//   2. waits for the job's producer event (cuEventQuery spin, exactly what
//      torch's event.synchronize() does for these non-blocking-sync events);
//   3. stores the slot's generation into the shared ready_generation table
//      (the connect-time snapshot in handle_ready reads the same array) and
//      posts one inline RDMA_WRITE_WITH_IMM ready record per active session
//      on its pull queue pair (signaled, wr_id = the queue pair's tag, the
//      ring index as the immediate);
//   4. reaps the shared send completion queue.
//
// The thread never touches a Python object and never takes the GIL.  Queue
// pair handles arrive from Python at connect (add_session) and leave before
// destruction (remove_session, called by fence() before ibv_destroy_qp); the
// session table's mutex makes that removal atomic with the publish step.
// Fail-stop: a post failure or an error completion disables the session here
// and is reported through the error callback (a ctypes closure that
// reacquires the GIL and fences the session, off the hot path); an
// event-wait error skips the job (the slot stays unready, as the Python
// loop does).
//
// Verbs are reached through dt_rdma_verbs.cpp's entry points, passed in at
// create (the dt_ring_worker.cpp pattern): post_write_imm_inline for the
// records and dt_rdma_poll with a zero timeout for reaping, so this library
// builds without the InfiniBand headers and links only libdl and pthread.
// CUDA is reached through the driver API (dlopen).
//
// Stamps (dt_native_probe.h, set once by dt_target_sender_set_probe): per job
// SENDER_TAKEN / SENDER_EVENT_DONE / per-session SENDER_POSTED, and one
// SENDER_CQE per reaped completion; the Target scheduler drains them into
// target_ready_publish / target_ready_cqe trace events while profiling.

#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <string.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#include "dt_native_probe.h"

namespace {

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUevent_st* CUevent;

struct CudaApi {
  CUresult (*cuInit)(unsigned);
  CUresult (*cuDeviceGet)(CUdevice*, int);
  CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
  CUresult (*cuDevicePrimaryCtxRelease)(CUdevice);
  CUresult (*cuCtxSetCurrent)(CUcontext);
  CUresult (*cuEventQuery)(CUevent);
};

// dt_rdma_verbs.cpp entry points (ctypes passes their addresses at create).
typedef int (*PostWriteImmInlineFn)(void* qp, uint64_t wr_id, const void* data,
                                    uint32_t length, uint64_t remote_addr,
                                    uint32_t rkey, uint32_t imm);
typedef int (*PollFn)(void* cq, int max, uint64_t timeout_ns, uint64_t* wr_id,
                      uint32_t* status, uint32_t* opcode, uint32_t* imm,
                      uint32_t* qpn);

constexpr CUresult kNotReady = 600;  // CUDA_ERROR_NOT_READY

bool load_cuda(CudaApi* api) {
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
  LOAD(cuEventQuery, "cuEventQuery");
#undef LOAD
  return true;
}

// dt_rdma.py's wire constants; the ready record layout itself is owned there
// (_READY_RECORD_FIELDS) and ported below by encode_ready_record, with the
// byte-parity test test_dt_target_sender.py guarding the port.
constexpr uint64_t kReadyRingDepth = 1024;  // READY_RING_DEPTH
constexpr uint32_t kReadyRecordBytes = 32;  // READY_RECORD_BYTES
constexpr uint32_t kRingSlots = 1024;       // queued ready jobs
constexpr uint32_t kPollBatch = 64;
// Idle wakeup period: completions must keep reaping with no jobs arriving.
constexpr auto kIdleWait = std::chrono::milliseconds(1);

// <QQQII little-endian: seq, pool-generation tag, source-session tag, slot
// id, slot generation (dt_rdma.py _READY_RECORD).
void encode_ready_record(uint8_t* out, uint64_t seq, uint64_t pool_tag,
                         uint64_t session_tag, uint32_t slot_id,
                         uint32_t slot_generation) {
  memcpy(out, &seq, 8);
  memcpy(out + 8, &pool_tag, 8);
  memcpy(out + 16, &session_tag, 8);
  memcpy(out + 24, &slot_id, 4);
  memcpy(out + 28, &slot_generation, 4);
}

struct Job {
  uint64_t event = 0;
  uint64_t enqueued_ns = 0;
  uint32_t slot = 0;
  uint32_t generation = 0;
};

struct Session {
  void* qp = nullptr;
  uint64_t tag = 0;
  uint64_t ready_addr = 0;
  uint32_t ready_rkey = 0;
  uint64_t ready_seq = 0;
  bool active = false;
  bool failed = false;
};

struct Sender {
  CudaApi api{};
  CUdevice device = 0;
  CUcontext ctx = nullptr;
  PostWriteImmInlineFn post = nullptr;
  PollFn poll = nullptr;
  void* cq = nullptr;
  uint32_t* ready_generation = nullptr;
  uint64_t slot_count = 0;
  uint64_t pool_tag = 0;
  uint64_t session_tag = 0;

  Job ring[kRingSlots];
  std::atomic<uint64_t> head{0};  // producer (the scheduler thread)
  std::atomic<uint64_t> tail{0};  // this thread
  std::atomic<uint64_t> done{0};  // last fully processed ticket

  std::mutex sessions_mutex;
  std::vector<Session> sessions;

  std::atomic<bool> stop{false};
  std::atomic<bool> sleeping{false};
  std::mutex mutex;
  std::condition_variable wake;
  std::thread thread;

  std::atomic<dt_probe::Probe*> probe{nullptr};

  // Error report, invoked with no sender lock held: (arg, tag, code, status);
  // tag 0 marks a job-level failure (the event wait), code 1 a post failure,
  // code 2 an error completion, code 3 the event wait itself.
  void (*error_cb)(void*, uint64_t, int32_t, int32_t) = nullptr;
  void* error_arg = nullptr;

  std::atomic<uint64_t> jobs{0}, posts{0}, cqes{0};
  std::atomic<uint64_t> post_errors{0}, cqe_errors{0}, event_errors{0};
};

Session* find_session(Sender* s, uint64_t tag) {
  for (Session& session : s->sessions)
    if (session.tag == tag) return &session;
  return nullptr;
}

void serve(Sender* s) {
  s->api.cuCtxSetCurrent(s->ctx);
  // Failures to report through the callback, outside every sender lock.
  struct Failure {
    uint64_t tag;
    int32_t code, status;
  };
  std::vector<Failure> failures;
  uint64_t wr_id[kPollBatch];
  uint32_t status[kPollBatch], opcode[kPollBatch], imm[kPollBatch], qpn[kPollBatch];
  while (true) {
    dt_probe::Probe* p = s->probe.load(std::memory_order_acquire);
    bool progress = false;
    uint64_t tail = s->tail.load(std::memory_order_relaxed);
    while (tail != s->head.load(std::memory_order_acquire)) {
      const Job job = s->ring[tail % kRingSlots];
      const uint64_t ticket = tail + 1;
      // Stamps key by (slot, generation): unique per publication.
      const uint64_t key = ((uint64_t)job.slot << 32) | job.generation;
      dt_probe::stamp(p, dt_probe::SENDER_TAKEN, -1, key, job.enqueued_ns);
      CUresult rc = 0;
      while (!s->stop.load(std::memory_order_relaxed) &&
             (rc = s->api.cuEventQuery((CUevent)job.event)) == kNotReady) {
        sched_yield();
      }
      dt_probe::stamp(p, dt_probe::SENDER_EVENT_DONE, -1, key, job.enqueued_ns);
      if (rc != 0 && !s->stop.load(std::memory_order_relaxed)) {
        // The slot stays unready (the Python loop logs and skips it too).
        s->event_errors.fetch_add(1, std::memory_order_relaxed);
        failures.push_back({0, 3, rc});
      } else if (rc == 0) {
        uint8_t record[kReadyRecordBytes];
        std::lock_guard<std::mutex> lock(s->sessions_mutex);
        if (job.slot < s->slot_count) s->ready_generation[job.slot] = job.generation;
        for (Session& session : s->sessions) {
          if (!session.active || session.failed) continue;
          session.ready_seq += 1;
          const uint32_t index = (uint32_t)(session.ready_seq % kReadyRingDepth);
          encode_ready_record(record, session.ready_seq, s->pool_tag,
                              s->session_tag, job.slot, job.generation);
          const int prc = s->post(session.qp, session.tag, record,
                                  kReadyRecordBytes,
                                  session.ready_addr + index * kReadyRecordBytes,
                                  session.ready_rkey, index);
          if (prc != 0) {
            session.failed = true;
            s->post_errors.fetch_add(1, std::memory_order_relaxed);
            failures.push_back({session.tag, 1, prc});
          } else {
            s->posts.fetch_add(1, std::memory_order_relaxed);
            dt_probe::stamp(p, dt_probe::SENDER_POSTED, -1, key,
                            (session.tag << 32) | index);
          }
        }
      }
      s->jobs.fetch_add(1, std::memory_order_relaxed);
      s->done.store(ticket, std::memory_order_release);
      s->tail.store(++tail, std::memory_order_release);
      progress = true;
    }
    const int n = s->poll(s->cq, kPollBatch, 0, wr_id, status, opcode, imm, qpn);
    for (int i = 0; i < n; ++i) {
      s->cqes.fetch_add(1, std::memory_order_relaxed);
      dt_probe::stamp(p, dt_probe::SENDER_CQE, -1, wr_id[i],
                      ((uint64_t)status[i] << 32) | imm[i]);
      if (status[i] != 0) {  // != IBV_WC_SUCCESS
        std::lock_guard<std::mutex> lock(s->sessions_mutex);
        if (Session* session = find_session(s, wr_id[i])) session->failed = true;
        s->cqe_errors.fetch_add(1, std::memory_order_relaxed);
        failures.push_back({wr_id[i], 2, (int32_t)status[i]});
      }
    }
    if (n > 0) progress = true;
    for (const Failure& failure : failures) {
      if (s->error_cb)
        s->error_cb(s->error_arg, failure.tag, failure.code, failure.status);
    }
    failures.clear();
    if (s->stop.load(std::memory_order_relaxed) &&
        tail == s->head.load(std::memory_order_acquire)) {
      return;
    }
    if (progress) continue;
    std::unique_lock<std::mutex> lock(s->mutex);
    s->sleeping.store(true);
    s->wake.wait_for(lock, kIdleWait, [&] {
      return s->stop.load() || tail != s->head.load();
    });
    s->sleeping.store(false);
  }
}

}  // namespace

extern "C" {

void* dt_target_sender_create(int device_index, void* cq, void* post_fn,
                              void* poll_fn, uint32_t* ready_generation,
                              uint64_t slot_count, uint64_t pool_tag,
                              uint64_t session_tag) {
  Sender* s = new (std::nothrow) Sender();
  if (s == nullptr) return nullptr;
  CUdevice dev;
  if (!load_cuda(&s->api) || s->api.cuInit(0) != 0 ||
      s->api.cuDeviceGet(&dev, device_index) != 0 ||
      s->api.cuDevicePrimaryCtxRetain(&s->ctx, dev) != 0) {
    delete s;
    return nullptr;
  }
  s->device = dev;
  s->cq = cq;
  s->post = (PostWriteImmInlineFn)post_fn;
  s->poll = (PollFn)poll_fn;
  s->ready_generation = ready_generation;
  s->slot_count = slot_count;
  s->pool_tag = pool_tag;
  s->session_tag = session_tag;
  s->thread = std::thread(serve, s);
  return s;
}

int dt_target_sender_set_probe(void* handle, void* probe) {
  Sender* s = static_cast<Sender*>(handle);
  dt_probe::Probe* expected = nullptr;
  return s->probe.compare_exchange_strong(
             expected, static_cast<dt_probe::Probe*>(probe),
             std::memory_order_release)
             ? 0
             : -1;
}

void dt_target_sender_set_error_callback(
    void* handle, void (*callback)(void*, uint64_t, int32_t, int32_t),
    void* arg) {
  Sender* s = static_cast<Sender*>(handle);
  s->error_arg = arg;
  std::atomic_thread_fence(std::memory_order_release);
  s->error_cb = callback;
}

// Register a connected pull queue pair; returns 0, or -1 for a duplicate tag.
int dt_target_sender_add_session(void* handle, uint64_t tag, void* qp,
                                 uint64_t ready_addr, uint32_t ready_rkey) {
  Sender* s = static_cast<Sender*>(handle);
  std::lock_guard<std::mutex> lock(s->sessions_mutex);
  if (find_session(s, tag) != nullptr) return -1;
  Session session;
  session.tag = tag;
  session.qp = qp;
  session.ready_addr = ready_addr;
  session.ready_rkey = ready_rkey;
  s->sessions.push_back(session);
  return 0;
}

void dt_target_sender_set_active(void* handle, uint64_t tag, int active) {
  Sender* s = static_cast<Sender*>(handle);
  std::lock_guard<std::mutex> lock(s->sessions_mutex);
  if (Session* session = find_session(s, tag)) session->active = active != 0;
}

// Unregister a queue pair; called by fence() before its ibv_destroy_qp, so
// this thread can never post into a destroyed queue pair.
void dt_target_sender_remove_session(void* handle, uint64_t tag) {
  Sender* s = static_cast<Sender*>(handle);
  std::lock_guard<std::mutex> lock(s->sessions_mutex);
  for (size_t i = 0; i < s->sessions.size(); ++i) {
    if (s->sessions[i].tag == tag) {
      s->sessions.erase(s->sessions.begin() + i);
      return;
    }
  }
}

// Queue one ready job; returns its ticket (>= 1), or 0 once the sender is
// stopping.  Blocks while the ring is full (the consumer retires a job
// within microseconds, so a full ring is backpressure, not a steady state).
uint64_t dt_target_sender_submit(void* handle, uint32_t slot,
                                 uint32_t generation, uint64_t event,
                                 uint64_t enqueued_ns) {
  Sender* s = static_cast<Sender*>(handle);
  const uint64_t head = s->head.load(std::memory_order_relaxed);
  while (head - s->tail.load(std::memory_order_acquire) >= kRingSlots) {
    if (s->stop.load(std::memory_order_relaxed)) return 0;
    sched_yield();
  }
  Job& job = s->ring[head % kRingSlots];
  job.event = event;
  job.enqueued_ns = enqueued_ns;
  job.slot = slot;
  job.generation = generation;
  s->head.store(head + 1, std::memory_order_seq_cst);
  if (s->sleeping.load(std::memory_order_seq_cst)) {
    std::lock_guard<std::mutex> lock(s->mutex);
    s->wake.notify_one();
  }
  return head + 1;
}

// The last ticket whose event wait and posts finished (acquire); jobs are
// processed in order, so every ticket up to it is done.
uint64_t dt_target_sender_done(void* handle) {
  return static_cast<Sender*>(handle)->done.load(std::memory_order_acquire);
}

// {jobs, posts, completions, post errors, completion errors, event errors}.
void dt_target_sender_stats(void* handle, uint64_t* out) {
  Sender* s = static_cast<Sender*>(handle);
  out[0] = s->jobs.load(std::memory_order_relaxed);
  out[1] = s->posts.load(std::memory_order_relaxed);
  out[2] = s->cqes.load(std::memory_order_relaxed);
  out[3] = s->post_errors.load(std::memory_order_relaxed);
  out[4] = s->cqe_errors.load(std::memory_order_relaxed);
  out[5] = s->event_errors.load(std::memory_order_relaxed);
}

int dt_target_sender_destroy(void* handle) {
  Sender* s = static_cast<Sender*>(handle);
  if (s == nullptr) return 0;
  s->stop.store(true);
  {
    std::lock_guard<std::mutex> lock(s->mutex);
    s->wake.notify_one();
  }
  s->thread.join();
  s->api.cuDevicePrimaryCtxRelease(s->device);
  const uint64_t errors =
      s->post_errors.load() + s->cqe_errors.load() + s->event_errors.load();
  delete s;
  return errors ? -1 : 0;
}

// Test hook (no CUDA, no RDMA): one ready record image, as posted; returns
// its byte count.  Byte-identical to dt_rdma.encode_ready_record by test.
uint64_t dt_target_sender_encode_record(uint8_t* out, uint64_t seq,
                                        uint64_t pool_tag,
                                        uint64_t session_tag,
                                        uint32_t slot_id,
                                        uint32_t slot_generation) {
  encode_ready_record(out, seq, pool_tag, session_tag, slot_id,
                      slot_generation);
  return kReadyRecordBytes;
}

}  // extern "C"
