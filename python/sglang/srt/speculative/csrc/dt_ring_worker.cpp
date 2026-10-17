// DT V2 Draft-side ring worker: Target -> owner-Draft descriptor rings.
//
// One native thread per Draft process polls every attached host ring. For each
// published entry it packs the request's hidden rows into the current Draft
// landing buffer and queues, on its own CUDA stream and without the Python
// GIL: a wait on the Target's interprocess producer event, the copy from the
// mapped Target source allocation, and a cuStreamWriteValue32 of the entry seq
// into the ring's INPUT_READY word.  No resident SMs are used: the copy runs on
// a copy engine and the publication is a stream memory operation.  CUDA is
// reached through the driver API (dlopen); the library links only libdl and
// pthread.
//
// Landing buffers are shared by all rings so one Draft compute batch can mix
// Targets.  The Draft main loop seals the buffer it takes a batch from (the
// worker moves to a free one) and frees a buffer once every entry placed in it
// has been released; both calls take the worker lock, so a buffer is never
// reused while a placed entry is unconsumed.
//
// The Python side (dt_direct_ring.py) owns every allocation and mapping and
// keeps them alive until the worker is destroyed (stream drained).

#include <dlfcn.h>
#include <sched.h>
#include <stdint.h>
#include <string.h>

#include <atomic>
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
struct CUipcEventHandle {
  char reserved[64];
};

constexpr uint32_t kEntryHasPull = 1u;
constexpr uint64_t kEventHandleOffset = 64;

struct Api {
  CUresult (*cuInit)(unsigned);
  CUresult (*cuDeviceGet)(CUdevice*, int);
  CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
  CUresult (*cuDevicePrimaryCtxRelease)(CUdevice);
  CUresult (*cuCtxSetCurrent)(CUcontext);
  CUresult (*cuStreamCreateWithPriority)(CUstream*, unsigned, int);
  CUresult (*cuCtxGetStreamPriorityRange)(int*, int*);
  CUresult (*cuStreamDestroy)(CUstream);
  CUresult (*cuStreamSynchronize)(CUstream);
  CUresult (*cuStreamWaitEvent)(CUstream, CUevent, unsigned);
  CUresult (*cuMemcpyAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
  CUresult (*cuStreamWriteValue32)(CUstream, CUdeviceptr, uint32_t, unsigned);
  CUresult (*cuIpcOpenEventHandle)(CUevent*, CUipcEventHandle);
  CUresult (*cuEventDestroy)(CUevent);
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
  LOAD(cuCtxGetStreamPriorityRange, "cuCtxGetStreamPriorityRange");
  LOAD(cuStreamDestroy, "cuStreamDestroy_v2");
  LOAD(cuStreamSynchronize, "cuStreamSynchronize");
  LOAD(cuStreamWaitEvent, "cuStreamWaitEvent");
  LOAD(cuMemcpyAsync, "cuMemcpyAsync");
  LOAD(cuStreamWriteValue32, "cuStreamWriteValue32_v2");
  LOAD(cuIpcOpenEventHandle, "cuIpcOpenEventHandle");
  LOAD(cuEventDestroy, "cuEventDestroy_v2");
#undef LOAD
  return true;
}

// Ring entry, written by the Target CPU with seq stored last.  Bytes 64..127
// hold the producer event's IPC handle; the task payload starts at 128.
struct EntryHeader {
  uint32_t seq;
  uint32_t flags;
  uint64_t source_offset;  // bytes from the mapped source allocation base
  uint32_t nbytes;
  uint32_t source_slot_id;
  uint32_t slot_generation;
  uint32_t payload_len;
  uint32_t epoch;
  uint32_t reserved[7];
};
static_assert(sizeof(EntryHeader) == 64, "ring entry header must be 64 bytes");

struct Ring {
  int id = 0;
  const uint8_t* entries = nullptr;  // host address of entry 0
  uint64_t entry_bytes = 0;
  uint32_t capacity = 0;
  CUdeviceptr ready_words = 0;       // device address of the INPUT_READY words
  CUdeviceptr source_base = 0;       // mapped Target source allocation
  uint64_t source_length = 0;
  int32_t* placement_buffer = nullptr;   // per entry: landing buffer index
  uint64_t* placement_offset = nullptr;  // per entry: byte offset in that buffer
  uint32_t next_seq = 1;
  bool detached = false;
};

struct Worker {
  Api api{};
  CUdevice device = 0;
  int cpu = -1;
  CUcontext ctx = nullptr;
  CUstream stream = nullptr;
  // Landing buffers (all equal size), shared by every ring.
  CUdeviceptr landing_base = 0;
  uint64_t landing_bytes = 0;
  uint64_t row_bytes = 1;
  std::vector<uint8_t> landing_free;    // 1 = free
  std::vector<uint8_t> landing_sealed;  // 1 = no more placements
  std::vector<uint64_t> landing_placed; // entries placed since last claim
  int current = -1;
  uint64_t cursor = 0;
  std::mutex lock;
  std::vector<Ring*> rings;
  std::atomic<bool> stop{false};
  std::atomic<int> error{0};  // first CUresult (>0) or protocol error (<0)
  std::atomic<int> error_ring{-1};
  std::atomic<uint64_t> entries_done{0};
  std::atomic<uint64_t> landing_stalls{0};
  std::thread thread;
};

void fail(Worker* w, int code, int ring_id) {
  int expected = 0;
  if (w->error.compare_exchange_strong(expected, code)) w->error_ring.store(ring_id);
}

// Choose where `nbytes` lands; -1 when every buffer is busy (caller stalls).
int place_locked(Worker* w, uint64_t nbytes, uint64_t* offset) {
  if (w->current >= 0 && !w->landing_sealed[w->current] && w->cursor + nbytes <= w->landing_bytes) {
    *offset = w->cursor;
    return w->current;
  }
  if (w->current >= 0) w->landing_sealed[w->current] = 1;  // abandoned: drains, then frees
  w->current = -1;
  for (size_t k = 0; k < w->landing_free.size(); ++k) {
    if (!w->landing_free[k]) continue;
    w->landing_free[k] = 0;
    w->landing_sealed[k] = 0;
    w->landing_placed[k] = 0;
    w->current = (int)k;
    w->cursor = 0;
    *offset = 0;
    return w->current;
  }
  return -1;
}

bool serve_entry(Worker* w, Ring* r) {
  const uint8_t* slot = r->entries + (uint64_t)(r->next_seq % r->capacity) * r->entry_bytes;
  const volatile uint32_t* seq_word = reinterpret_cast<const volatile uint32_t*>(slot);
  if (*seq_word != r->next_seq) return false;
  std::atomic_thread_fence(std::memory_order_acquire);
  EntryHeader h;
  memcpy(&h, slot, sizeof(h));
  const uint32_t index = r->next_seq % r->capacity;
  r->placement_buffer[index] = -1;
  r->placement_offset[index] = 0;
  if (h.flags & kEntryHasPull) {
    if (h.nbytes == 0 || h.nbytes % w->row_bytes || h.nbytes > w->landing_bytes ||
        h.source_offset + h.nbytes > r->source_length) {
      fail(w, -2, r->id);
      return false;
    }
    uint64_t offset = 0;
    const int k = place_locked(w, h.nbytes, &offset);
    if (k < 0) {
      w->landing_stalls.fetch_add(1, std::memory_order_relaxed);
      return false;  // retry once the main loop frees a landing buffer
    }
    // The Target records a fresh interprocess event per publication; the wait
    // binds to that record, and destroying our handle after enqueueing it is
    // allowed (the driver keeps the event until the wait retires).
    CUipcEventHandle event_handle;
    memcpy(&event_handle, slot + kEventHandleOffset, sizeof(event_handle));
    CUevent producer = nullptr;
    CUresult rc = w->api.cuIpcOpenEventHandle(&producer, event_handle);
    if (rc == 0) rc = w->api.cuStreamWaitEvent(w->stream, producer, 0);
    if (producer) w->api.cuEventDestroy(producer);
    if (rc == 0)
      rc = w->api.cuMemcpyAsync(w->landing_base + (uint64_t)k * w->landing_bytes + offset,
                                r->source_base + h.source_offset, h.nbytes, w->stream);
    if (rc != 0) {
      fail(w, rc, r->id);
      return false;
    }
    w->cursor = offset + h.nbytes;
    w->landing_placed[k] += 1;
    r->placement_buffer[index] = k;
    r->placement_offset[index] = offset;
  }
  // Default stream memop semantics order this write after the copy above;
  // the placement words were stored by this thread before it.
  CUresult rc = w->api.cuStreamWriteValue32(w->stream, r->ready_words + 4ull * index,
                                            r->next_seq, 0);
  if (rc != 0) {
    fail(w, rc, r->id);
    return false;
  }
  // Wire seq 0 is the never-written value; the Python side skips it too.
  if (++r->next_seq == 0) r->next_seq = 1;
  w->entries_done.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void run(Worker* w) {
  if (w->cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(w->cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
  }
  if (w->api.cuCtxSetCurrent(w->ctx) != 0) {
    fail(w, -3, -1);
    return;
  }
  while (!w->stop.load(std::memory_order_relaxed) && w->error.load() == 0) {
    bool progressed = false;
    {
      std::lock_guard<std::mutex> guard(w->lock);
      for (Ring* r : w->rings) {
        if (r->detached) continue;
        // Bounded per ring per pass so one busy Target cannot starve another.
        for (int i = 0; i < 64 && serve_entry(w, r); ++i) progressed = true;
      }
    }
    if (!progressed) sched_yield();
  }
}

}  // namespace

extern "C" {

// Returns an opaque worker or nullptr.  `landing_base` holds `landing_count`
// equal buffers of `landing_bytes`; entries are packed in `row_bytes` units.
void* dt_ring_worker_create(int device, int cpu, uint64_t landing_base, uint64_t landing_bytes,
                            uint32_t landing_count, uint64_t row_bytes) {
  if (!landing_count || !landing_bytes || !row_bytes || landing_bytes % row_bytes) return nullptr;
  Worker* w = new Worker();
  if (!load_api(&w->api) || w->api.cuInit(0) != 0) {
    delete w;
    return nullptr;
  }
  CUdevice dev;
  if (w->api.cuDeviceGet(&dev, device) != 0 ||
      w->api.cuDevicePrimaryCtxRetain(&w->ctx, dev) != 0) {
    delete w;
    return nullptr;
  }
  w->device = dev;
  w->cpu = cpu;
  w->landing_base = landing_base;
  w->landing_bytes = landing_bytes;
  w->row_bytes = row_bytes;
  w->landing_free.assign(landing_count, 1);
  w->landing_sealed.assign(landing_count, 0);
  w->landing_placed.assign(landing_count, 0);
  int least = 0, greatest = 0;
  w->api.cuCtxSetCurrent(w->ctx);
  w->api.cuCtxGetStreamPriorityRange(&least, &greatest);
  // CU_STREAM_NON_BLOCKING: never serialize with the legacy default stream.
  if (w->api.cuStreamCreateWithPriority(&w->stream, 0x1, greatest) != 0) {
    w->api.cuDevicePrimaryCtxRelease(dev);
    delete w;
    return nullptr;
  }
  return w;
}

int dt_ring_worker_attach(void* handle, int ring_id, uint64_t entries_host, uint64_t entry_bytes,
                          uint32_t capacity, uint64_t ready_words_device, uint64_t source_base,
                          uint64_t source_length, uint64_t placement_buffer,
                          uint64_t placement_offset, uint32_t first_seq) {
  Worker* w = static_cast<Worker*>(handle);
  if (!w || !entries_host || entry_bytes < 128 || !capacity || !first_seq || !placement_buffer ||
      !placement_offset)
    return -1;
  Ring* r = new Ring();
  r->id = ring_id;
  r->entries = reinterpret_cast<const uint8_t*>(entries_host);
  r->entry_bytes = entry_bytes;
  r->capacity = capacity;
  r->ready_words = ready_words_device;
  r->source_base = source_base;
  r->source_length = source_length;
  r->placement_buffer = reinterpret_cast<int32_t*>(placement_buffer);
  r->placement_offset = reinterpret_cast<uint64_t*>(placement_offset);
  r->next_seq = first_seq;
  std::lock_guard<std::mutex> guard(w->lock);
  w->rings.push_back(r);
  return 0;
}

// Stop serving a ring; returns its next unserved seq.  Its queued copies may
// still run: drain with dt_ring_worker_sync before unmapping anything.
uint32_t dt_ring_worker_detach(void* handle, int ring_id) {
  Worker* w = static_cast<Worker*>(handle);
  std::lock_guard<std::mutex> guard(w->lock);
  for (Ring* r : w->rings)
    if (r->id == ring_id && !r->detached) {
      r->detached = true;
      return r->next_seq;
    }
  return 0;
}

// No further entries land in buffer k; returns how many were placed in it.
uint64_t dt_ring_worker_seal(void* handle, uint32_t k) {
  Worker* w = static_cast<Worker*>(handle);
  std::lock_guard<std::mutex> guard(w->lock);
  if (k >= w->landing_free.size()) return 0;
  w->landing_sealed[k] = 1;
  if (w->current == (int)k) w->current = -1;
  return w->landing_placed[k];
}

// Free buffer k if it is sealed and all `released` placed entries are done.
int dt_ring_worker_try_free(void* handle, uint32_t k, uint64_t released) {
  Worker* w = static_cast<Worker*>(handle);
  std::lock_guard<std::mutex> guard(w->lock);
  if (k >= w->landing_free.size() || w->landing_free[k]) return -1;
  if (!w->landing_sealed[k] || w->current == (int)k || w->landing_placed[k] != released)
    return 0;
  w->landing_free[k] = 1;
  w->landing_sealed[k] = 0;
  w->landing_placed[k] = 0;
  return 1;
}

int dt_ring_worker_start(void* handle) {
  Worker* w = static_cast<Worker*>(handle);
  if (w->thread.joinable()) return -1;
  w->thread = std::thread(run, w);
  return 0;
}

int dt_ring_worker_error(void* handle, int* ring_id) {
  Worker* w = static_cast<Worker*>(handle);
  if (ring_id) *ring_id = w->error_ring.load();
  return w->error.load();
}

uint64_t dt_ring_worker_entries_done(void* handle) {
  return static_cast<Worker*>(handle)->entries_done.load();
}

uint64_t dt_ring_worker_landing_stalls(void* handle) {
  return static_cast<Worker*>(handle)->landing_stalls.load();
}

int dt_ring_worker_sync(void* handle) {
  Worker* w = static_cast<Worker*>(handle);
  w->api.cuCtxSetCurrent(w->ctx);
  return w->api.cuStreamSynchronize(w->stream);
}

// Stop the thread, drain its stream, release everything it owns.
int dt_ring_worker_destroy(void* handle) {
  Worker* w = static_cast<Worker*>(handle);
  if (!w) return 0;
  w->stop.store(true);
  if (w->thread.joinable()) w->thread.join();
  w->api.cuCtxSetCurrent(w->ctx);
  int rc = w->api.cuStreamSynchronize(w->stream);
  w->api.cuStreamDestroy(w->stream);
  w->api.cuDevicePrimaryCtxRelease(w->device);
  for (Ring* r : w->rings) delete r;
  delete w;
  return rc;
}

}  // extern "C"
