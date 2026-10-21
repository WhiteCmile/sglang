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
#include <deque>
#include <memory>
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

// §7.1 kernel-published record (DT_RING_PUBLISH=1): same 4096-byte stride,
// u64 seq released last by the GPU, version word 2 at byte 12 (old entries
// keep 0 there because the source allocation is < 4 GiB).  The producer
// event handle sits at byte 3976, written by the Target CPU before launch.
constexpr uint32_t kRecordVersion2 = 2;
constexpr uint64_t kRecordV2MarkerBXor = 0x9E3779B97F4A7C15ull;
constexpr uint64_t kRecordV2EventHandleOffset = 3976;
constexpr uint32_t kRecordV2MaxReq = 56;

struct RecordV2Header {
  uint64_t seq;
  uint32_t flags;
  uint32_t version;
  uint64_t owner_epoch;
  uint32_t source_slot_id;
  uint32_t slot_generation;
  uint64_t source_byte_offset;
  uint64_t nbytes;
  uint32_t row_count;
  uint32_t row_bytes;
  uint32_t request_count;
  uint32_t block_size;
  uint64_t hidden_produced_ns;
  uint64_t marker_a;
  uint64_t reserved[6];
};
static_assert(sizeof(RecordV2Header) == 128, "v2 record header must be 128 bytes");

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
  uint64_t slot_nbytes = 0;   // registered slot geometry (0: unfenced)
  uint32_t slot_count = 0;
  uint64_t allocation_offset = 0;          // registered source base offset
  const uint32_t* slot_generations = nullptr;  // per-slot current generation
  uint64_t pool_tag = 0;  // registered pool-generation tag for v2 records
  uint64_t next_seq = 1;
  bool detached = false;
  // Consumed-credit state: the worker snapshots every entry it serves into
  // the mailbox before releasing its ring slot back to the Target.
  uint32_t* header = nullptr;      // ring header words (H_CONSUMED lives at 5)
  uint8_t* mailbox = nullptr;      // capacity mailbox slots of mailbox_stride
  uint64_t mailbox_stride = 0;
  const uint64_t* mail_ack = nullptr;  // last mailbox seq the main loop parsed
  void* ledger = nullptr;          // native rid ledger (v2 records), or null
};

// Mailbox slot layout (all offsets in bytes).  The worker publishes the seq
// tag last; the main loop acks by storing the consumed seq into mail_ack.
constexpr uint64_t kMailSeqTag = 0;
constexpr uint64_t kMailFormat = 8;     // 1 = v1 entry, 2 = v2 record
constexpr uint64_t kMailCount = 12;     // v2 request_count / v1 payload_len
constexpr uint64_t kMailTokTotal = 16;  // v2 compacted token count
constexpr uint64_t kMailHeader = 24;    // up to 128 B (v2 header; v1 uses 64)
constexpr uint64_t kMailReqs = 152;     // 56 * 32 B request table
constexpr uint64_t kMailTokens = 1944;  // 512 * 4 B compacted token ids
constexpr uint64_t kMailV1Payload = 3992;  // up to 3968 B (entry stride 4096)
constexpr uint64_t kMailRids = 7960;    // 56 * 256 B resolved rid strings
constexpr uint64_t kMailMarkerB = 22296;   // v2 tail marker (8 B)
constexpr uint64_t kMailEvent = 22304;     // v2 producer event handle (64 B)
constexpr uint64_t kMailDropped = 22368;   // per-request dropped bitmask (8 B)
constexpr uint64_t kMailMinStride = kMailDropped + 8;

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

void advance_seq(Ring* r) {
  // Wire seq 0 is the never-written value; the Python side skips it too.
  if (!(uint32_t)++r->next_seq) ++r->next_seq;
}

// The mailbox slot for the next lap may still be unparsed: stop serving
// until the main loop's ack comes within one lap.  Same backpressure shape
// as a full landing set.
bool mailbox_free(Ring* r) {
  return r->mailbox == nullptr ||
         r->next_seq - *r->mail_ack < r->capacity;
}

// Snapshot one served entry into its mailbox slot and seal it.  After this
// returns the ring entry may be reused by the Target at any time.
void mailbox_publish(Ring* r, uint32_t index, uint32_t format, uint32_t count,
                     uint32_t tok_total, const uint8_t* header, uint32_t header_bytes,
                     const uint8_t* reqs, uint32_t reqs_bytes,
                     const uint8_t* tokens, uint32_t tokens_bytes,
                     const uint8_t* payload, uint32_t payload_bytes) {
  uint8_t* mb = r->mailbox + (uint64_t)index * r->mailbox_stride;
  *reinterpret_cast<uint32_t*>(mb + kMailFormat) = format;
  *reinterpret_cast<uint32_t*>(mb + kMailCount) = count;
  *reinterpret_cast<uint32_t*>(mb + kMailTokTotal) = tok_total;
  memcpy(mb + kMailHeader, header, header_bytes);
  if (reqs_bytes) memcpy(mb + kMailReqs, reqs, reqs_bytes);
  if (tokens_bytes) memcpy(mb + kMailTokens, tokens, tokens_bytes);
  if (payload_bytes) memcpy(mb + kMailV1Payload, payload, payload_bytes);
  std::atomic_thread_fence(std::memory_order_release);
  *reinterpret_cast<volatile uint64_t*>(mb + kMailSeqTag) = r->next_seq;
}

extern "C" int dt_rid_ledger_resolve(void* handle, uint64_t hash, char* out,
                                     uint32_t out_len);
extern "C" void dt_rid_ledger_reader_open(void* handle);
extern "C" void dt_rid_ledger_reader_close(void* handle);

// Resolve every request's rid into the mailbox rid region.  Returns false
// when any hash is not registered yet; the caller retries next pass.  A
// released hash (tombstone) is not retried: its request is marked dropped so
// the record is consumed without ever blocking the ring behind it.  The
// dropped bitmask is reported through `dropped_out` (and sealed into the
// mailbox by the caller).
bool mailbox_resolve_rids(Ring* r, const uint8_t* reqs, uint32_t request_count,
                          uint64_t* dropped_out = nullptr) {
  if (r->ledger == nullptr) return false;
  uint8_t* mb = r->mailbox + (uint64_t)(r->next_seq % r->capacity) * r->mailbox_stride;
  uint64_t dropped = 0;
  for (uint32_t i = 0; i < request_count; ++i) {
    uint64_t hash;
    memcpy(&hash, reqs + (uint64_t)i * 32, 8);
    char* out = reinterpret_cast<char*>(mb + kMailRids + (uint64_t)i * 256);
    int len = dt_rid_ledger_resolve(r->ledger, hash, out, 256);
    if (len == -5) {
      out[0] = 0;
      dropped |= 1ull << i;
      continue;
    }
    if (len <= 0) return false;
  }
  if (dropped_out != nullptr) *dropped_out = dropped;
  else if (dropped) *reinterpret_cast<uint64_t*>(mb + kMailDropped) = dropped;
  return true;
}

// Default stream memop semantics order the ready write after the pull copy;
// the placement words were stored by this thread before it.  The consumed
// word advances here — the entry's content is already sealed in the mailbox,
// so the Target may reuse the ring entry from this point on.
bool finish_entry(Worker* w, Ring* r) {
  CUresult rc = w->api.cuStreamWriteValue32(
      w->stream, r->ready_words + 4ull * (r->next_seq % r->capacity),
      (uint32_t)r->next_seq, 0);
  if (rc != 0) {
    fail(w, rc, r->id);
    return false;
  }
  if (r->header != nullptr) r->header[5] = (uint32_t)r->next_seq;
  advance_seq(r);
  w->entries_done.fetch_add(1, std::memory_order_relaxed);
  return true;
}

// Enqueue the landing copy for a pull entry whose geometry already checked
// out; returns false when every landing buffer is busy (caller retries).
bool enqueue_pull(Worker* w, Ring* r, uint32_t index, uint64_t source_offset,
                  uint64_t nbytes, const uint8_t* event_handle_bytes) {
  uint64_t offset = 0;
  const int k = place_locked(w, nbytes, &offset);
  if (k < 0) {
    w->landing_stalls.fetch_add(1, std::memory_order_relaxed);
    return false;  // retry once the main loop frees a landing buffer
  }
  // The Target records a fresh interprocess event per publication; the wait
  // binds to that record, and destroying our handle after enqueueing it is
  // allowed (the driver keeps the event until the wait retires).
  CUipcEventHandle event_handle;
  memcpy(&event_handle, event_handle_bytes, sizeof(event_handle));
  CUevent producer = nullptr;
  CUresult rc = w->api.cuIpcOpenEventHandle(&producer, event_handle);
  if (rc == 0) rc = w->api.cuStreamWaitEvent(w->stream, producer, 0);
  if (producer) w->api.cuEventDestroy(producer);
  if (rc == 0)
    rc = w->api.cuMemcpyAsync(w->landing_base + (uint64_t)k * w->landing_bytes + offset,
                              r->source_base + source_offset, nbytes, w->stream);
  if (rc != 0) {
    fail(w, rc, r->id);
    return false;
  }
  w->cursor = offset + nbytes;
  w->landing_placed[k] += 1;
  r->placement_buffer[index] = k;
  r->placement_offset[index] = offset;
  return true;
}

bool serve_entry_v2(Worker* w, Ring* r) {
  const uint8_t* slot = r->entries + (uint64_t)(r->next_seq % r->capacity) * r->entry_bytes;
  const volatile uint64_t* seq_word = reinterpret_cast<const volatile uint64_t*>(slot);
  if (*seq_word != r->next_seq) return false;
  std::atomic_thread_fence(std::memory_order_acquire);
  RecordV2Header h;
  memcpy(&h, slot, sizeof(h));
  uint64_t marker_b;
  memcpy(&marker_b, slot + 3968, 8);
  const uint32_t index = r->next_seq % r->capacity;
  r->placement_buffer[index] = -1;
  r->placement_offset[index] = 0;
  // Geometry and marker validation; the seq release already proved the
  // record was fully written by the kernel.  row_count is bounded by the
  // landing capacity through nbytes (token ids are compacted separately).
  if (h.version != kRecordVersion2 || !(h.flags & kEntryHasPull) ||
      h.marker_a != ~r->next_seq || marker_b != (r->next_seq ^ kRecordV2MarkerBXor) ||
      h.request_count > kRecordV2MaxReq ||
      h.nbytes == 0 || h.nbytes != (uint64_t)h.row_count * h.row_bytes ||
      h.row_bytes != w->row_bytes || h.nbytes > w->landing_bytes ||
      h.source_byte_offset + h.nbytes > r->source_length) {
    fail(w, -4, r->id);
    return false;
  }
  // Registered-source fences: the record must name the attached pool
  // generation, the slot's current generation, and a slot-aligned range
  // fully inside its registered slot.
  bool geometry_ok =
      !r->slot_nbytes ||
      (h.source_byte_offset >= r->allocation_offset &&
       h.source_byte_offset - r->allocation_offset ==
           (uint64_t)h.source_slot_id * r->slot_nbytes &&
       h.source_byte_offset + h.nbytes <=
           r->allocation_offset + (uint64_t)(h.source_slot_id + 1) * r->slot_nbytes);
  if (h.reserved[1] != r->pool_tag ||
      (r->slot_count && h.source_slot_id >= r->slot_count) || !geometry_ok ||
      (r->slot_generations && h.source_slot_id < r->slot_count &&
       r->slot_generations[h.source_slot_id] != h.slot_generation)) {
    fail(w, -5, r->id);
    return false;
  }
  if (!mailbox_free(r)) return false;
  if (r->mailbox != nullptr) {
    const uint8_t* reqs = slot + 128;              // request table
    const uint8_t* tokens = slot + 128 + 56 * 32;  // compacted token ids
    // Resolve rids first: an unregistered hash must not seal the mailbox.
    uint64_t dropped = 0;
    if (!mailbox_resolve_rids(r, reqs, h.request_count, &dropped)) return false;
    uint32_t token_total = 0;
    for (uint32_t i = 0; i < h.request_count; ++i) {
      uint32_t cnt;
      memcpy(&cnt, reqs + (uint64_t)i * 32 + 20, 4);  // token_count field
      token_total += cnt;
    }
    uint8_t* mb = r->mailbox + (uint64_t)index * r->mailbox_stride;
    memcpy(mb + kMailMarkerB, slot + 3968, 8);
    memcpy(mb + kMailEvent, slot + kRecordV2EventHandleOffset, 64);
    *reinterpret_cast<uint64_t*>(mb + kMailDropped) = dropped;
    mailbox_publish(r, index, 2, h.request_count, token_total, slot,
                    sizeof(h), reqs, h.request_count * 32, tokens,
                    token_total * 4, nullptr, 0);
    if (dropped == (h.request_count >= 64 ? ~0ull : ((1ull << h.request_count) - 1))) {
      // Every request is released: no landing copy is needed — the draft
      // will never read this wave, so the bare INPUT_READY is the true
      // read proof, and the ring moves on instead of stalling.
      return true;
    }
  }
  return enqueue_pull(w, r, index, h.source_byte_offset, h.nbytes,
                      slot + kRecordV2EventHandleOffset);
}

bool serve_entry_v1(Worker* w, Ring* r) {
  const uint8_t* slot = r->entries + (uint64_t)(r->next_seq % r->capacity) * r->entry_bytes;
  const volatile uint32_t* seq_word = reinterpret_cast<const volatile uint32_t*>(slot);
  if (*seq_word != (uint32_t)r->next_seq) return false;
  std::atomic_thread_fence(std::memory_order_acquire);
  EntryHeader h;
  memcpy(&h, slot, sizeof(h));
  // A stale slot from a lap 2**32 entries back aliases the low seq word.
  if (h.epoch != (uint32_t)(r->next_seq >> 32)) return false;
  const uint32_t index = r->next_seq % r->capacity;
  r->placement_buffer[index] = -1;
  r->placement_offset[index] = 0;
  if (!mailbox_free(r)) return false;
  if (r->mailbox != nullptr) {
    if (h.payload_len > r->entry_bytes - 128) {
      fail(w, -2, r->id);
      return false;
    }
    mailbox_publish(r, index, 1, h.payload_len, 0, slot, sizeof(h), nullptr, 0,
                    nullptr, 0, slot + 128, h.payload_len);
  }
  if (h.flags & kEntryHasPull) {
    if (h.nbytes == 0 || h.nbytes % w->row_bytes || h.nbytes > w->landing_bytes ||
        h.source_offset + h.nbytes > r->source_length) {
      fail(w, -2, r->id);
      return false;
    }
    if (!enqueue_pull(w, r, index, h.source_offset, h.nbytes,
                      slot + kEventHandleOffset))
      return false;
  }
  return finish_entry(w, r);
}

bool serve_entry(Worker* w, Ring* r) {
  const uint8_t* slot = r->entries + (uint64_t)(r->next_seq % r->capacity) * r->entry_bytes;
  // Publication order is seq last for both formats, and bytes 0..3 hold the
  // low 32 bits of seq in both.  Acquire the publication word first; only
  // then is the format discriminator at byte 12 stable.  Reading the
  // discriminator first could route a record whose publication landed
  // between the two reads into the wrong parser — publishing INPUT_READY
  // without pulling anything.
  const volatile uint64_t* seq_word = reinterpret_cast<const volatile uint64_t*>(slot);
  const uint64_t first_word = *seq_word;
  if ((uint32_t)first_word != (uint32_t)r->next_seq) return false;
  std::atomic_thread_fence(std::memory_order_acquire);
  const volatile uint32_t* disc = reinterpret_cast<const volatile uint32_t*>(slot + 12);
  if (*disc == kRecordVersion2) {
    // v2 publishes the full u64 seq; anything else is a torn or stale read.
    if (first_word != r->next_seq) return false;
    if (serve_entry_v2(w, r)) return finish_entry(w, r);
    return false;
  }
  // v1: the high half of the first word is the flags field (0 or 1).
  if (first_word >> 32 > 1) return false;
  return serve_entry_v1(w, r);
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
      // Register for the whole pass so writer-side reclamation can only fire
      // between passes, when this thread provably holds no entry.  All rings
      // share one ledger in production; opening per ring stays correct if
      // that ever changes.
      for (Ring* r : w->rings) {
        if (r->detached) continue;
        if (r->ledger != nullptr) dt_rid_ledger_reader_open(r->ledger);
        // Bounded per ring per pass so one busy Target cannot starve another.
        for (int i = 0; i < 64 && serve_entry(w, r); ++i) progressed = true;
        if (r->ledger != nullptr) dt_rid_ledger_reader_close(r->ledger);
      }
    }
    if (!progressed) sched_yield();
  }
}

}  // namespace

extern "C" {

// Process-wide native rid_hash -> rid ledger (§7.1).  The Draft main loop is
// the only writer (request admission/release); the worker thread resolves
// rids while serving v2 records without taking the GIL or a mutex.  Cells
// are plain atomic pointers to immutable entries, so reads are lock-free
// MOV loads (no shared_ptr spinlock); entries are reclaimed only at
// destroy(), which the engine calls after the worker thread has stopped.
// Removal publishes a released tombstone in place instead of shifting
// probes, so readers never stop early, and insertion probes the whole chain
// for a live key before reusing a dead cell, so idempotent re-registration
// can never duplicate an entry.
struct RidLedger {
  struct Entry {
    uint64_t key;  // rid_hash + 1
    char rid[256];
    bool released;  // tombstone: consumed-and-dropped by the worker, never stalled
  };
  std::deque<std::atomic<const Entry*>> slots;
  // Replaced entries retire here until NO reader is inside any read
  // (reader-count RCU): the worker registers around each serve pass and
  // resolve() registers around its probe, so readers==0 is a proven
  // quiescent point.  No epoch protocol — retention is bounded by reads
  // actually in flight.
  std::deque<const Entry*> retired;
  std::atomic<int> readers{0};
  // Observability only: counts reader registrations (the reviewers' tooling
  // prints it); reclamation keys on `readers`, not on this counter.
  std::atomic<uint64_t> reader_epoch{0};
};

// Free retired entries once no reader can be inside a read.  Called by the
// writer only; the deque is writer-owned, so no lock is needed.  The fence
// is load-bearing: it publishes the replacement store that retired these
// entries before the readers-check, so a reader whose guard increment
// follows the check still observes the new cell content (never a freed one);
// a reader whose increment precedes the store is visible to the check.
static void ledger_reclaim(RidLedger* l) {
  std::atomic_thread_fence(std::memory_order_seq_cst);
  if (l->readers.load(std::memory_order_acquire) != 0) return;
  while (!l->retired.empty()) {
    delete l->retired.front();
    l->retired.pop_front();
  }
}

void* dt_rid_ledger_create(uint32_t capacity_pow2) {
  if (!capacity_pow2 || (capacity_pow2 & (capacity_pow2 - 1))) return nullptr;
  RidLedger* l = new RidLedger();
  for (uint32_t i = 0; i < capacity_pow2; ++i) l->slots.emplace_back();
  return l;
}

int dt_rid_ledger_put(void* handle, uint64_t hash, const char* rid) {
  RidLedger* l = static_cast<RidLedger*>(handle);
  if (!l || !rid || !*rid || strlen(rid) >= sizeof(RidLedger::Entry::rid)) return -3;
  const uint64_t key = hash + 1;
  const uint64_t mask = l->slots.size() - 1;
  // Phase 1: search the whole chain for a live key (duplicate/collision
  // check) while remembering the first dead cell.  Tombstones count as
  // reusable space, so a table full of them still admits.
  size_t first_dead = SIZE_MAX;
  for (size_t i = key & mask;; i = (i + 1) & mask) {
    const RidLedger::Entry* cur = l->slots[i].load(std::memory_order_acquire);
    if (cur == nullptr) {
      if (first_dead == SIZE_MAX) first_dead = i;
      break;
    }
    if (!cur->released && cur->key == key)
      return strcmp(cur->rid, rid) == 0 ? 0 : -1;  // hash collision
    if (cur->released && first_dead == SIZE_MAX) first_dead = i;
    if (i == ((key - 1) & mask)) break;
  }
  if (first_dead == SIZE_MAX) return -2;  // genuinely full of live entries
  RidLedger::Entry* body = new RidLedger::Entry();
  body->key = key;
  memcpy(body->rid, rid, strlen(rid) + 1);
  body->released = false;
  const RidLedger::Entry* prev =
      l->slots[first_dead].load(std::memory_order_acquire);
  l->slots[first_dead].store(body, std::memory_order_release);
  if (prev != nullptr) l->retired.push_back(prev);
  ledger_reclaim(l);
  return 0;
}

int dt_rid_ledger_remove(void* handle, uint64_t hash, const char* rid) {
  RidLedger* l = static_cast<RidLedger*>(handle);
  if (!l || !rid) return -3;
  const uint64_t key = hash + 1;
  const uint64_t mask = l->slots.size() - 1;
  for (size_t i = key & mask;; i = (i + 1) & mask) {
    const RidLedger::Entry* cur = l->slots[i].load(std::memory_order_acquire);
    if (cur == nullptr) return 0;
    if (!cur->released && cur->key == key && strcmp(cur->rid, rid) == 0) {
      RidLedger::Entry* stone = new RidLedger::Entry(*cur);
      stone->released = true;
      l->slots[i].store(stone, std::memory_order_release);
      l->retired.push_back(cur);
      ledger_reclaim(l);
      return 0;
    }
    if (i == ((key - 1) & mask)) return 0;
  }
}

struct LedgerReadGuard {
  RidLedger* l;
  explicit LedgerReadGuard(RidLedger* ledger) : l(ledger) {
    if (l) l->readers.fetch_add(1, std::memory_order_acq_rel);
  }
  ~LedgerReadGuard() {
    if (l) l->readers.fetch_sub(1, std::memory_order_acq_rel);
  }
};

// >0: rid length.  0: unknown hash (caller may retry: registration may lag).
// -5: released hash (tombstone; the record must be consumed and dropped).
// The guard registers this thread for the probe's duration, so the writer's
// reclamation can never free an entry mid-read.
int dt_rid_ledger_resolve(void* handle, uint64_t hash, char* out, uint32_t out_len) {
  RidLedger* l = static_cast<RidLedger*>(handle);
  if (!l || !out || !out_len) return -3;
  LedgerReadGuard guard(l);
  const uint64_t key = hash + 1;
  const uint64_t mask = l->slots.size() - 1;
  for (size_t i = key & mask;; i = (i + 1) & mask) {
    const RidLedger::Entry* cur = l->slots[i].load(std::memory_order_acquire);
    if (cur == nullptr) return 0;
    if (cur->key == key) {
      if (cur->released) return -5;
      const size_t len = strlen(cur->rid);
      if (len + 1 > out_len) return -4;
      memcpy(out, cur->rid, len + 1);
      return (int)len;
    }
    if (i == ((key - 1) & mask)) return 0;
  }
}

void dt_rid_ledger_reader_open(void* handle) {
  RidLedger* l = static_cast<RidLedger*>(handle);
  if (l) {
    l->readers.fetch_add(1, std::memory_order_acq_rel);
    l->reader_epoch.fetch_add(1, std::memory_order_relaxed);
  }
}

void dt_rid_ledger_reader_close(void* handle) {
  RidLedger* l = static_cast<RidLedger*>(handle);
  if (l) l->readers.fetch_sub(1, std::memory_order_acq_rel);
}

void dt_rid_ledger_destroy(void* handle) {
  RidLedger* l = static_cast<RidLedger*>(handle);
  if (!l) return;
  for (auto& cell : l->slots) delete cell.load();
  for (const RidLedger::Entry* e : l->retired) delete e;
  delete l;
}

}  // extern "C"

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
                          uint64_t placement_offset, uint64_t first_seq,
                          uint64_t slot_nbytes, uint32_t slot_count, uint64_t pool_tag,
                          uint64_t slot_generations_host, uint64_t allocation_offset,
                          uint64_t header_host, uint64_t mailbox_base,
                          uint64_t mailbox_stride, uint64_t mail_ack_host,
                          void* ledger) {
  Worker* w = static_cast<Worker*>(handle);
  if (!w || !entries_host || entry_bytes < 128 || !capacity || !first_seq || !placement_buffer ||
      !placement_offset)
    return -1;
  if (mailbox_base &&
      (mailbox_stride < kMailMinStride || !header_host || !mail_ack_host))
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
  r->slot_nbytes = slot_nbytes;
  r->slot_count = slot_count;
  r->allocation_offset = allocation_offset;
  r->slot_generations = reinterpret_cast<const uint32_t*>(slot_generations_host);
  r->pool_tag = pool_tag;
  r->next_seq = first_seq;
  r->header = reinterpret_cast<uint32_t*>(header_host);
  r->mailbox = reinterpret_cast<uint8_t*>(mailbox_base);
  r->mailbox_stride = mailbox_stride;
  r->mail_ack = reinterpret_cast<const uint64_t*>(mail_ack_host);
  r->ledger = ledger;
  std::lock_guard<std::mutex> guard(w->lock);
  w->rings.push_back(r);
  return 0;
}

// Stop serving a ring; returns its next unserved seq.  Its queued copies may
// still run: drain with dt_ring_worker_sync before unmapping anything.
uint64_t dt_ring_worker_detach(void* handle, int ring_id) {
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
