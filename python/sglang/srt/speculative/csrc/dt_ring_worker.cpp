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
//
// Cross-host rings (issue #198) take the same path through a different data
// mover.  The Target writes v1 entries into a pinned Draft ring with RDMA
// WRITE_WITH_IMM (issue #182); the worker consumes the arrival completions,
// applies the local fences plus the per-slot generation table the Target
// mirrors onto the ring, waits for the slot's ready record (the Target's
// producer finished), places the rows in the same landing buffers, and posts
// the GPUDirect RDMA READ itself.  One cuFlushGPUDirectRDMAWrites covers every
// READ completed in a pass, after which the INPUT_READY words are stored.
// Verbs are reached through dt_rdma_verbs.cpp's entry points, passed in at
// attach; this library still links no RDMA library.
//
// Native ingress (Draft redesign D6, design §6, DT_DRAFT_INGRESS=native):
// after serving an entry the worker also publishes per-row READY entries
// (fixed 128-byte ReadyEntry) on an SPSC queue to the scheduler.  v2 records
// are parsed here; v1 entries (cross host, same-host non-v2 waves) become one
// V1_PAYLOAD entry naming the mailbox slot the scheduler decodes (until
// Target item T4).  Ownership (§4.3): the worker writes the queue head, the
// token arena, the copy-event pool, landing gen/placed/sealed and the mailbox
// pins; the scheduler writes only ready_consumed_seq, and the landing
// releases arrive on their own SPSC queue.  Every reclaim keys on
// ready_consumed_seq: a queue slot, a token span, a copy event and a v1
// mailbox slot are reused only after the scheduler passed every entry naming
// them.

#include <dlfcn.h>
#include <sched.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
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
  CUresult (*cuEventCreate)(CUevent*, unsigned);
  CUresult (*cuEventRecord)(CUevent, CUstream);
  CUresult (*cuFlushGPUDirectRDMAWrites)(int, int);
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
  LOAD(cuEventCreate, "cuEventCreate");
  LOAD(cuEventRecord, "cuEventRecord");
  LOAD(cuFlushGPUDirectRDMAWrites, "cuFlushGPUDirectRDMAWrites");
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

// One v2 request-table row (dt_direct_ring.RingRecordV2Request field order).
struct RecordV2Request {
  uint64_t rid_hash;
  uint32_t round_id;
  uint32_t attempt;
  uint32_t token_offset;
  uint32_t token_count;
  uint32_t bonus_token;
  uint32_t seq_len;
};
static_assert(sizeof(RecordV2Request) == 32, "v2 request row must be 32 bytes");

// One destination-region triple (dt_direct_ring.TargetRingWriter.write_destinations).
struct DestTriple {
  uint32_t slot_id;
  uint32_t generation;
  uint64_t nonce;
};
static_assert(sizeof(DestTriple) == 16, "destination triple must be 16 bytes");

// §6.2 READY entry.  The Python mirror is built from dt_ring_ready_field, so
// this declaration is the only copy of the layout.
#define DT_READY_FIELDS(X)                                                   \
  X(seq)            /* READY seq (1-based queue generation) */                 \
  X(kind)           /* kReady* kind, | kReadyDestKnown */                      \
  X(ring_id)                                                                   \
  X(record_seq)     /* ring seq of the served entry */                         \
  X(position)       /* request index in a v2 record */                         \
  X(req_handle)     /* rid-ledger cell */                                      \
  X(req_gen)        /* its generation */                                       \
  X(round_id) X(attempt)                                                       \
  X(seq_len)        /* committed_kv_len */                                     \
  X(token_count) X(bonus_token)                                                \
  X(landing_k)      /* landing buffer, -1: none */                             \
  X(landing_gen)                                                               \
  X(row_first)      /* first landing row of the request */                     \
  X(dest_slot) X(dest_gen) X(dest_nonce)                                       \
  X(copy_event_idx) /* kNoEvent: none (cross host, no pull) */                 \
  X(copy_event_gen)                                                            \
  X(token_at)       /* token-arena index of the token ids (monotonic) */       \
  X(transfer_hash)  /* blake2b-64 of "target:rid:round:attempt" */             \
  X(served_ns) X(ready_ns)

struct ReadyEntry {
  uint64_t seq;
  uint32_t kind;
  int32_t ring_id;
  uint64_t record_seq;
  uint32_t position, req_handle, req_gen, round_id, attempt, seq_len, token_count, bonus_token;
  int32_t landing_k;
  uint32_t landing_gen;
  uint64_t row_first;
  uint32_t dest_slot, dest_gen;
  uint64_t dest_nonce;
  uint32_t copy_event_idx, copy_event_gen;
  uint64_t token_at, transfer_hash, served_ns, ready_ns;
};
static_assert(sizeof(ReadyEntry) == 128, "READY entry must be 128 bytes");

#define DT_READY_CONSTANTS(X)                                       \
  X(ROW, 1)         /* a live v2 request, parsed natively */          \
  X(DROPPED, 2)     /* a released v2 request with a known destination */ \
  X(V1_PAYLOAD, 3)  /* a v1 entry: decode its mailbox payload */        \
  X(DEST_KNOWN, 0x100) /* kind flag: dest_* read from the region */    \
  X(NO_EVENT, 0xFFFFFFFFu)
#define DT_READY_ENUM(name, value) constexpr uint32_t kReady_##name = value;
DT_READY_CONSTANTS(DT_READY_ENUM)
#undef DT_READY_ENUM

uint64_t fnv1a64(const char* s) {
  uint64_t h = 0xCBF29CE484222325ull;
  for (; *s; ++s) h = (h ^ (uint8_t)*s) * 0x100000001B3ull;
  return h;
}

// Unkeyed BLAKE2b with an 8-byte digest (RFC 7693), read little-endian:
// dt_candidate_p2p.transfer_id_hash.
uint64_t blake2b64(const uint8_t* data, size_t len) {
  static const uint64_t iv[8] = {
      0x6A09E667F3BCC908ull, 0xBB67AE8584CAA73Bull, 0x3C6EF372FE94F82Bull,
      0xA54FF53A5F1D36F1ull, 0x510E527FADE682D1ull, 0x9B05688C2B3E6C1Full,
      0x1F83D9ABFB41BD6Bull, 0x5BE0CD19137E2179ull};
  static const uint8_t sigma[12][16] = {
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
      {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
      {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
      {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
      {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
      {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
      {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
      {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
      {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
      {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
      {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};
  auto rotr = [](uint64_t x, int n) { return (x >> n) | (x << (64 - n)); };
  uint64_t h[8];
  memcpy(h, iv, sizeof(h));
  h[0] ^= 0x01010000ull ^ 8;  // digest length 8, no key
  size_t done = 0;
  do {
    const size_t take = len - done > 128 ? 128 : len - done;
    const bool last = done + take == len;
    uint64_t m[16] = {};
    memcpy(m, data + done, take);  // little-endian host
    done += take;
    uint64_t v[16];
    memcpy(v, h, sizeof(h));
    memcpy(v + 8, iv, sizeof(iv));
    v[12] ^= done;
    if (last) v[14] = ~v[14];
    for (int round = 0; round < 12; ++round) {
      const uint8_t* s = sigma[round];
      auto g = [&](int a, int b, int c, int d, uint64_t x, uint64_t y) {
        v[a] += v[b] + x; v[d] = rotr(v[d] ^ v[a], 32);
        v[c] += v[d];     v[b] = rotr(v[b] ^ v[c], 24);
        v[a] += v[b] + y; v[d] = rotr(v[d] ^ v[a], 16);
        v[c] += v[d];     v[b] = rotr(v[b] ^ v[c], 63);
      };
      g(0, 4, 8, 12, m[s[0]], m[s[1]]);
      g(1, 5, 9, 13, m[s[2]], m[s[3]]);
      g(2, 6, 10, 14, m[s[4]], m[s[5]]);
      g(3, 7, 11, 15, m[s[6]], m[s[7]]);
      g(0, 5, 10, 15, m[s[8]], m[s[9]]);
      g(1, 6, 11, 12, m[s[10]], m[s[11]]);
      g(2, 7, 8, 13, m[s[12]], m[s[13]]);
      g(3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[i + 8];
  } while (done < len);
  return h[0];
}

// dt_rdma_verbs.cpp entry points (ctypes passes their addresses at attach).
typedef int (*PostReadFn)(void* qp, uint64_t wr_id, uint64_t local_addr, uint32_t length,
                          uint32_t lkey, uint64_t remote_addr, uint32_t rkey);
typedef int (*PostRecvFn)(void* qp, uint64_t wr_id, int count);
typedef int (*PollFn)(void* cq, int max, uint64_t timeout_ns, uint64_t* wr_id,
                      uint32_t* status, uint32_t* opcode, uint32_t* imm, uint32_t* qpn);

// Attach-time description of one cross-host ring; every field is a u64 so the
// ctypes mirror (dt_direct_ring.RemoteRingSpec) is a flat list in this order.
struct RemoteRingSpec {
  uint64_t post_read, post_recv, poll;
  // Task-record leg: arrivals on recv_cq; the worker's READs go out on the
  // same queue pair and complete on send_cq.
  uint64_t record_qp, record_recv_cq, record_send_cq, record_tag;
  uint64_t table_host;   // per-slot generation table the Target mirrors
  uint64_t read_window;  // READs outstanding at once (the QP's send depth)
  uint64_t landing_lkey;
  // Remote feature pool: READ address = pool_addr + offset - allocation_offset.
  uint64_t pool_addr, pool_rkey, pool_allocation_offset, slot_nbytes, slot_count;
  // Ready-record leg (#165): the worker is its only consumer.  ready_qp == 0
  // when the Target serves no feature pool (token-only rounds).
  uint64_t ready_qp, ready_recv_cq, ready_tag, ready_ring_host, ready_depth,
      ready_first_seq, ready_table_host, pool_tag, session_tag;
  uint64_t drain_timeout_ns;  // destroy() waits this long for posted READs
  // Values dt_rdma owns, passed here instead of copied: the work-completion
  // status/opcode the worker accepts, the cuFlushGPUDirectRDMAWrites
  // arguments, and the ready-record layout (stride, then each field's byte
  // offset and width as dt_rdma's record struct defines it).
  uint64_t wc_success, wc_recv_with_imm, flush_target, flush_scope;
  uint64_t ready_record_bytes;
  uint64_t ready_seq_at, ready_seq_bytes, ready_pool_tag_at, ready_pool_tag_bytes,
      ready_session_tag_at, ready_session_tag_bytes, ready_slot_id_at, ready_slot_id_bytes,
      ready_slot_generation_at, ready_slot_generation_bytes;
};

// Every ready-record field the spec names fits a u64 and lies in the record.
bool ready_layout_ok(const RemoteRingSpec& s) {
  const uint64_t fields[][2] = {
      {s.ready_seq_at, s.ready_seq_bytes},
      {s.ready_pool_tag_at, s.ready_pool_tag_bytes},
      {s.ready_session_tag_at, s.ready_session_tag_bytes},
      {s.ready_slot_id_at, s.ready_slot_id_bytes},
      {s.ready_slot_generation_at, s.ready_slot_generation_bytes},
  };
  for (const auto& f : fields)
    if (!f[1] || f[1] > 8 || f[0] + f[1] > s.ready_record_bytes) return false;
  return true;
}

// One little-endian field of a ready record, at the offset/width the spec names.
uint64_t ready_field(const uint8_t* record, uint64_t at, uint64_t width) {
  uint64_t value = 0;
  memcpy(&value, record + at, width);
  return value;
}

// One served cross-host entry, published strictly in seq order.
struct Flight {
  uint64_t seq;
  uint32_t index;
  uint32_t slot_id;
  uint32_t slot_generation;
  bool read;  // a READ was posted for it
  bool done;  // its READ completed (or it had none)
  uint32_t landing_gen;  // native ingress: its READY entry's placement
  uint64_t served_ns;
};

struct Remote {
  RemoteRingSpec spec{};
  PostReadFn post_read = nullptr;
  PostRecvFn post_recv = nullptr;
  PollFn poll = nullptr;
  uint32_t* ready_host = nullptr;        // INPUT_READY words (host memory)
  const uint32_t* table = nullptr;       // Target-mirrored slot generations
  uint32_t* ready_table = nullptr;       // slot -> highest ready generation
  uint64_t ready_expected = 1;           // next ready-record seq
  std::deque<uint32_t> arrived;          // immediates of unserved records
  std::deque<Flight> flights;            // served, not yet INPUT_READY
  uint64_t reads_outstanding = 0;
};

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
  uint64_t seen_seq = 0;  // last seq stamped RECORD_SEEN (probe only)
  bool detached = false;
  // Per-ring copy stream: entries from different Targets must not serialize.
  CUstream stream = nullptr;
  // Consumed-credit state: the worker snapshots every entry it serves into
  // the mailbox before releasing its ring slot back to the Target.
  uint32_t* header = nullptr;      // ring header words (H_CONSUMED lives at 5)
  uint8_t* mailbox = nullptr;      // capacity mailbox slots of mailbox_stride
  uint64_t mailbox_stride = 0;
  const uint64_t* mail_ack = nullptr;  // last mailbox seq the main loop parsed
  void* ledger = nullptr;          // native rid ledger (v2 records), or null
  std::unique_ptr<Remote> remote;  // set for a cross-host ring
  // Last seq this worker consumed (content snapshotted, ring slot released);
  // read by the scheduler for its rid eviction horizon.
  std::atomic<uint64_t> consumed_seq{0};
  // Native ingress only.  target_id: the transfer-hash prefix of this ring's
  // rows.  dest: the v2 destination region (§6.3).  mail_pin[index]: the
  // READY seq of the V1_PAYLOAD entry naming that mailbox slot (UINT64_MAX
  // while its READY entry is not yet published).
  char target_id[128] = {};
  const uint8_t* dest = nullptr;
  uint64_t dest_stride = 0, dest_seq_at = 0;
  std::vector<uint64_t> mail_pin;
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

// Native ingress state (D6).  The worker owns all of it except `consumed`
// (ready_consumed_seq, scheduler) and the release queue head (its producer).
struct Ingress {
  std::vector<ReadyEntry> queue;     // power of two
  std::atomic<uint64_t> published{0};  // worker: last READY seq stored
  std::atomic<uint64_t> consumed{0};   // scheduler: ready_consumed_seq
  std::vector<uint32_t> tokens;      // token arena; a span never wraps
  uint64_t token_head = 0, token_tail = 0;
  // One per published record, in READY order: what its last entry pins.
  struct Hold {
    uint64_t last_seq;
    uint64_t token_end;
    uint32_t event;
  };
  std::deque<Hold> holds;
  std::vector<CUevent> events;  // copy-event pool
  std::vector<uint32_t> event_gen;
  std::vector<uint32_t> free_events;
  // Landing releases (k, gen, placements), SPSC from one producer thread.
  struct Release {
    uint32_t k, gen;
    uint64_t count;
  };
  std::vector<Release> releases;  // power of two
  std::atomic<uint64_t> release_head{0};  // producer
  std::atomic<uint64_t> release_tail{0};  // worker
  void (*wake_set)(void*, uint32_t) = nullptr;  // dt_wake_set, or null
  void* wake = nullptr;
  uint32_t wake_bit = 0;
  // Per-record scratch of the v2 rid resolution.
  uint32_t handles[kRecordV2MaxReq], gens[kRecordV2MaxReq];
};

struct Worker {
  Api api{};
  CUdevice device = 0;
  int cpu = -1;
  CUcontext ctx = nullptr;
  CUstream stream = nullptr;
  int stream_priority = 0;  // highest priority reported by the device
  // Landing buffers (all equal size), shared by every ring.
  CUdeviceptr landing_base = 0;
  uint64_t landing_bytes = 0;
  uint64_t row_bytes = 1;
  std::vector<uint8_t> landing_free;    // 1 = free
  std::vector<uint8_t> landing_sealed;  // 1 = no more placements
  std::vector<uint64_t> landing_placed; // entries placed since last claim
  std::vector<uint64_t> landing_released;  // native ingress: placements released
  std::vector<uint32_t> landing_gen;    // bumped at every claim
  int current = -1;
  uint64_t cursor = 0;
  std::mutex lock;
  std::vector<Ring*> rings;
  std::atomic<bool> stop{false};
  std::atomic<int> error{0};  // first CUresult (>0) or protocol error (<0)
  std::atomic<int> error_ring{-1};
  std::atomic<uint64_t> entries_done{0};
  std::atomic<uint64_t> landing_stalls{0};
  dt_probe::Probe* probe = nullptr;  // set before start; null unless profiling
  std::unique_ptr<Ingress> ingress;  // set before start; null unless native ingress
  std::thread thread;
};

void stamp(Worker* w, dt_probe::Kind kind, const Ring* r, uint64_t seq, uint64_t arg = 0) {
  dt_probe::stamp(w->probe, kind, r->id, seq, arg);
}

// Landing placement as one stamp argument: buffer index above byte offset.
uint64_t placement_arg(int k, uint64_t offset) { return (uint64_t)k << 40 | offset; }

uint64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void fail(Worker* w, int code, int ring_id) {
  int expected = 0;
  if (w->error.compare_exchange_strong(expected, code)) w->error_ring.store(ring_id);
}

// Choose where `nbytes` lands; -1 when every buffer is busy (caller stalls).
// Native ingress free rule (§6.4): a buffer sealed by overflow frees once
// every placement in it was released.
void landing_free_if_released(Worker* w, size_t k) {
  if (!w->landing_sealed[k] || w->current == (int)k ||
      w->landing_released[k] != w->landing_placed[k])
    return;
  w->landing_free[k] = 1;
  w->landing_sealed[k] = 0;
  w->landing_placed[k] = 0;
  w->landing_released[k] = 0;
}

int place_locked(Worker* w, uint64_t nbytes, uint64_t* offset) {
  if (w->current >= 0 && !w->landing_sealed[w->current] && w->cursor + nbytes <= w->landing_bytes) {
    *offset = w->cursor;
    return w->current;
  }
  const int abandoned = w->current;
  if (abandoned >= 0) w->landing_sealed[abandoned] = 1;  // abandoned: drains, then frees
  w->current = -1;
  if (abandoned >= 0 && w->ingress) landing_free_if_released(w, abandoned);
  for (size_t k = 0; k < w->landing_free.size(); ++k) {
    if (!w->landing_free[k]) continue;
    w->landing_free[k] = 0;
    w->landing_sealed[k] = 0;
    w->landing_placed[k] = 0;
    w->landing_released[k] = 0;
    ++w->landing_gen[k];
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
// as a full landing set.  Under native ingress the worker acks its own
// mailbox: a v2 slot is free once parsed, a v1 slot once the scheduler's
// ready_consumed_seq passed the V1_PAYLOAD entry naming it.  The main
// loop's ack still gates while the Python consumer reads the mailbox
// (mail_ack set).
bool mailbox_free(Worker* w, Ring* r) {
  if (r->mailbox == nullptr) return true;
  if (r->mail_ack != nullptr && r->next_seq - *r->mail_ack >= r->capacity) return false;
  return r->mail_pin.empty() ||
         r->mail_pin[r->next_seq % r->capacity] <=
             w->ingress->consumed.load(std::memory_order_acquire);
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

extern "C" int dt_rid_ledger_lookup(void* handle, uint64_t hash, char* out,
                                    uint32_t out_len, uint32_t* cell, uint32_t* gen);
extern "C" void dt_rid_ledger_reader_open(void* handle);
extern "C" void dt_rid_ledger_reader_close(void* handle);

// Resolve every request's rid into `rids` (256 B per request).  Returns
// false when any hash is not registered yet; the caller retries next pass.
// A released hash (tombstone) is not retried: its request is marked dropped
// so the record is consumed without ever blocking the ring behind it (its
// rid is still written; the Python consumer ignores dropped rids).  With
// `handles`/`gens`, each request's ledger cell and generation are reported.
bool resolve_rids(void* ledger, const uint8_t* reqs, uint32_t request_count, uint8_t* rids,
                  uint64_t* dropped_out, uint32_t* handles, uint32_t* gens) {
  if (ledger == nullptr) return false;
  uint64_t dropped = 0;
  for (uint32_t i = 0; i < request_count; ++i) {
    uint64_t hash;
    memcpy(&hash, reqs + (uint64_t)i * 32, 8);
    uint32_t cell = 0, gen = 0;
    int len = dt_rid_ledger_lookup(ledger, hash, reinterpret_cast<char*>(rids + (uint64_t)i * 256),
                                   256, &cell, &gen);
    if (len == -5) dropped |= 1ull << i;
    else if (len <= 0) return false;
    if (handles) {
      handles[i] = cell;
      gens[i] = gen;
    }
  }
  *dropped_out = dropped;
  return true;
}

// §6.3: read one request's destination triple by seqlock (the validity word
// equals the record seq before and after).  False: not written yet.
bool read_dest(const uint8_t* dest, uint64_t seq_at, uint64_t seq, uint32_t position,
               DestTriple* out) {
  if (dest == nullptr) return false;
  const volatile uint64_t* word = reinterpret_cast<const volatile uint64_t*>(dest + seq_at);
  if (*word != seq) return false;
  std::atomic_thread_fence(std::memory_order_acquire);
  memcpy(out, dest + (uint64_t)position * sizeof(DestTriple), sizeof(DestTriple));
  std::atomic_thread_fence(std::memory_order_acquire);
  return *word == seq;
}

// §6.2: the READY entries of one validated v2 record, from its mailbox
// snapshot `mb` (request table, compacted token ids, resolved rids).  One
// ROW per live request, with its token ids appended to `tokens` (token_at
// relative to it); one DROPPED per released request whose destination is
// known and live.  `base` carries the ring, record and placement fields.
// Applies the field checks of the Python take (_accept_ring_record).
// Returns the entry count, or -4 for a malformed request.
int parse_v2_rows(const RecordV2Header& h, const uint8_t* mb, uint64_t dropped,
                  const uint32_t* handles, const uint32_t* gens, const char* target_id,
                  const uint8_t* dest, uint64_t dest_seq_at, const ReadyEntry& base,
                  uint64_t landing_row0, ReadyEntry* out, uint32_t* tokens, uint32_t* ntok) {
  int n = 0;
  uint64_t prefix = 0;  // compacted token position, dropped requests included
  *ntok = 0;
  for (uint32_t i = 0; i < h.request_count; ++i) {
    RecordV2Request q;
    memcpy(&q, mb + kMailReqs + (uint64_t)i * sizeof(q), sizeof(q));
    const char* rid = reinterpret_cast<const char*>(mb + kMailRids + (uint64_t)i * 256);
    const uint64_t first = prefix;
    prefix += q.token_count;  // u64: a forged count cannot wrap past the bound
    if (prefix > h.row_count || prefix > 512) return -4;
    const bool is_dropped = dropped >> i & 1;
    if (!is_dropped && (q.round_id < 1 || q.token_count == 0 || q.seq_len < q.token_count ||
                        (uint64_t)q.token_offset + q.token_count > h.row_count ||
                        fnv1a64(rid) != q.rid_hash))
      return -4;
    DestTriple d{};
    const bool known = read_dest(dest, dest_seq_at, h.seq, i, &d);
    if (is_dropped && (!known || (!d.slot_id && !d.generation && !d.nonce))) continue;
    ReadyEntry e = base;
    e.kind = (is_dropped ? kReady_DROPPED : kReady_ROW) | (known ? kReady_DEST_KNOWN : 0);
    e.position = i;
    e.req_handle = handles[i];
    e.req_gen = gens[i];
    e.round_id = q.round_id;
    e.attempt = q.attempt;
    e.seq_len = q.seq_len;
    e.bonus_token = q.bonus_token;
    e.row_first = landing_row0 + q.token_offset;
    e.dest_slot = d.slot_id;
    e.dest_gen = d.generation;
    e.dest_nonce = d.nonce;
    char transfer[512];
    const int len = snprintf(transfer, sizeof(transfer), "%s:%s:%u:%u", target_id, rid,
                             q.round_id, q.attempt);
    e.transfer_hash = blake2b64(reinterpret_cast<const uint8_t*>(transfer), (size_t)len);
    e.token_at = *ntok;
    if (!is_dropped) {  // a DROPPED entry carries no token ids
      e.token_count = q.token_count;
      memcpy(tokens + *ntok, mb + kMailTokens + (uint64_t)first * 4, (uint64_t)q.token_count * 4);
      *ntok += q.token_count;
    }
    out[n++] = e;
  }
  return n;
}

// The fixed fields of every READY entry of one served entry.
ReadyEntry ready_base(const Ring* r, uint64_t record_seq, uint64_t served_ns) {
  ReadyEntry e{};
  e.ring_id = r->id;
  e.record_seq = record_seq;
  e.landing_k = -1;
  e.copy_event_idx = kReady_NO_EVENT;
  e.served_ns = served_ns;
  return e;
}

// §6.5: a pooled event recorded on the ring stream behind the landing copy.
// A pool index is reused only after ready_consumed_seq passed its holder.
bool record_copy_event(Worker* w, Ring* r, uint32_t* index, uint32_t* gen) {
  Ingress& g = *w->ingress;
  uint32_t i;
  if (!g.free_events.empty()) {
    i = g.free_events.back();
    g.free_events.pop_back();
    ++g.event_gen[i];
  } else {
    CUevent event = nullptr;
    // CU_EVENT_DISABLE_TIMING: an ordering edge only.
    if (CUresult rc = w->api.cuEventCreate(&event, 0x2); rc != 0) {
      fail(w, rc, r->id);
      return false;
    }
    i = (uint32_t)g.events.size();
    g.events.push_back(event);
    g.event_gen.push_back(1);
  }
  if (CUresult rc = timed(w->probe, dt_probe::EVENT_RECORD,
                          [&] { return w->api.cuEventRecord(g.events[i], r->stream); });
      rc != 0) {
    fail(w, rc, r->id);
    return false;
  }
  *index = i;
  *gen = g.event_gen[i];
  return true;
}

// Publish one served entry's READY entries and wake the scheduler.  Their
// token_at is relative to `tokens` until here.  A full queue or arena is an
// invariant violation (both are sized above the resident landing rows), so
// it fails stop instead of stalling.  Returns the last READY seq, 0 on failure.
uint64_t ready_publish(Worker* w, Ring* r, ReadyEntry* e, uint32_t n, const uint32_t* tokens,
                       uint32_t ntok, uint32_t event) {
  Ingress& g = *w->ingress;
  const uint64_t capacity = g.queue.size(), arena = g.tokens.size();
  const uint64_t first = g.published.load(std::memory_order_relaxed);
  uint64_t at = g.token_head;
  if (at % arena + ntok > arena) at += arena - at % arena;
  if (first + n - g.consumed.load(std::memory_order_acquire) > capacity ||
      at + ntok - g.token_tail > arena) {
    fail(w, -8, r->id);
    return 0;
  }
  if (ntok) memcpy(&g.tokens[at % arena], tokens, (uint64_t)ntok * 4);
  g.token_head = at + ntok;
  const uint64_t now = now_ns();
  for (uint32_t i = 0; i < n; ++i) {
    e[i].seq = first + 1 + i;
    e[i].token_at += at;
    e[i].ready_ns = now;
    g.queue[(first + i) % capacity] = e[i];
  }
  g.published.store(first + n, std::memory_order_release);
  g.holds.push_back(Ingress::Hold{first + n, g.token_head, event});
  stamp(w, dt_probe::READY_PUBLISHED, r, e[0].record_seq, n);
  if (g.wake_set) g.wake_set(g.wake, g.wake_bit);
  return first + n;
}

// One V1_PAYLOAD entry for a served v1 entry at mailbox `index`, placed at
// (k, offset) or nowhere (k < 0).  Pins the mailbox slot until consumed.
bool publish_v1(Worker* w, Ring* r, uint64_t record_seq, uint32_t index, int k, uint32_t gen,
                uint64_t offset, uint64_t served_ns) {
  ReadyEntry e = ready_base(r, record_seq, served_ns);
  e.kind = kReady_V1_PAYLOAD;
  if (k >= 0) {
    e.landing_k = k;
    e.landing_gen = gen;
    e.row_first = offset / w->row_bytes;
    if (!r->remote && !record_copy_event(w, r, &e.copy_event_idx, &e.copy_event_gen))
      return false;
  }
  const uint64_t last = ready_publish(w, r, &e, 1, nullptr, 0, e.copy_event_idx);
  if (last) r->mail_pin[index] = last;
  return last != 0;
}

// Native ingress for one served v2 record at mailbox `index` (k < 0: no
// pull, every request dropped).
bool publish_v2(Worker* w, Ring* r, const RecordV2Header& h, uint32_t index, uint64_t dropped,
                int k, uint64_t offset, uint64_t served_ns) {
  Ingress& g = *w->ingress;
  const uint8_t* mb = r->mailbox + (uint64_t)index * r->mailbox_stride;
  ReadyEntry base = ready_base(r, h.seq, served_ns);
  if (k >= 0) {
    base.landing_k = k;
    base.landing_gen = w->landing_gen[k];
  }
  ReadyEntry entries[kRecordV2MaxReq];
  uint32_t tokens[512], ntok = 0;
  const int n = parse_v2_rows(h, mb, dropped, g.handles, g.gens, r->target_id,
                              r->dest ? r->dest + (uint64_t)index * r->dest_stride : nullptr,
                              r->dest_seq_at, base, offset / w->row_bytes, entries, tokens, &ntok);
  if (n < 0) {
    fail(w, n, r->id);
    return false;
  }
  if (n == 0) return true;
  uint32_t event = kReady_NO_EVENT, gen = 0;
  if (k >= 0 && !record_copy_event(w, r, &event, &gen)) return false;
  for (int i = 0; i < n; ++i) {
    entries[i].copy_event_idx = event;
    entries[i].copy_event_gen = gen;
  }
  return ready_publish(w, r, entries, (uint32_t)n, tokens, ntok, event) != 0;
}

// Pass start: retire what ready_consumed_seq passed, then apply landing
// releases (§4.3: the producer never writes landing state itself).  A
// release naming a stale generation or more placements than were made is a
// fail-stop violation (R7).
void ingress_reclaim(Worker* w) {
  Ingress& g = *w->ingress;
  const uint64_t consumed = g.consumed.load(std::memory_order_acquire);
  while (!g.holds.empty() && g.holds.front().last_seq <= consumed) {
    g.token_tail = g.holds.front().token_end;
    if (g.holds.front().event != kReady_NO_EVENT) g.free_events.push_back(g.holds.front().event);
    g.holds.pop_front();
  }
  const uint64_t head = g.release_head.load(std::memory_order_acquire);
  uint64_t tail = g.release_tail.load(std::memory_order_relaxed);
  for (; tail != head; ++tail) {
    const Ingress::Release rel = g.releases[tail % g.releases.size()];
    if (rel.k >= w->landing_free.size() || w->landing_free[rel.k] ||
        rel.gen != w->landing_gen[rel.k] ||
        w->landing_released[rel.k] + rel.count > w->landing_placed[rel.k]) {
      fail(w, -9, -1);
      break;
    }
    w->landing_released[rel.k] += rel.count;
    landing_free_if_released(w, rel.k);
  }
  g.release_tail.store(tail, std::memory_order_release);
}

// Default stream memop semantics order the ready write after the pull copy;
// the placement words were stored by this thread before it.  The consumed
// word advances here — the entry's content is already sealed in the mailbox,
// so the Target may reuse the ring entry from this point on.
bool finish_entry(Worker* w, Ring* r) {
  CUresult rc = timed(w->probe, dt_probe::WRITE_VALUE32, [&] {
    return w->api.cuStreamWriteValue32(
        r->stream, r->ready_words + 4ull * (r->next_seq % r->capacity),
        (uint32_t)r->next_seq, 0);
  });
  if (rc != 0) {
    fail(w, rc, r->id);
    return false;
  }
  stamp(w, dt_probe::INPUT_READY_ISSUED, r, r->next_seq);
  if (r->header != nullptr) r->header[5] = (uint32_t)r->next_seq;
  r->consumed_seq.store(r->next_seq, std::memory_order_release);
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
  stamp(w, dt_probe::PLACEMENT, r, r->next_seq, placement_arg(k, offset));
  CUevent producer = nullptr;
  CUresult rc = timed(w->probe, dt_probe::IPC_OPEN_EVENT,
                      [&] { return w->api.cuIpcOpenEventHandle(&producer, event_handle); });
  if (rc == 0)
    rc = timed(w->probe, dt_probe::STREAM_WAIT_EVENT,
               [&] { return w->api.cuStreamWaitEvent(r->stream, producer, 0); });
  if (producer)
    timed(w->probe, dt_probe::EVENT_DESTROY, [&] { return w->api.cuEventDestroy(producer); });
  if (rc == 0)
    rc = timed(w->probe, dt_probe::MEMCPY_ASYNC, [&] {
      return w->api.cuMemcpyAsync(w->landing_base + (uint64_t)k * w->landing_bytes + offset,
                                  r->source_base + source_offset, nbytes, r->stream);
    });
  if (rc != 0) {
    fail(w, rc, r->id);
    return false;
  }
  stamp(w, dt_probe::COPY_ENQUEUED, r, r->next_seq, nbytes);
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
  const uint64_t served_ns = w->ingress ? now_ns() : 0;
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
  // generation, the slot's current generation, and a row-aligned range fully
  // inside its registered slot.  A wave split across several records (one per
  // owner Draft group) names a subrange that starts past the slot base.
  const uint64_t slot_base =
      r->allocation_offset + (uint64_t)h.source_slot_id * r->slot_nbytes;
  bool geometry_ok =
      !r->slot_nbytes ||
      (h.source_byte_offset >= slot_base &&
       (h.source_byte_offset - slot_base) % h.row_bytes == 0 &&
       h.source_byte_offset + h.nbytes <=
           r->allocation_offset + (uint64_t)(h.source_slot_id + 1) * r->slot_nbytes);
  if (h.reserved[1] != r->pool_tag ||
      (r->slot_count && h.source_slot_id >= r->slot_count) || !geometry_ok ||
      (r->slot_generations && h.source_slot_id < r->slot_count &&
       r->slot_generations[h.source_slot_id] != h.slot_generation)) {
    fail(w, -5, r->id);
    return false;
  }
  if (!mailbox_free(w, r)) return false;
  uint64_t dropped = 0;
  if (r->mailbox != nullptr) {
    const uint8_t* reqs = slot + 128;              // request table
    const uint8_t* tokens = slot + 128 + 56 * 32;  // compacted token ids
    uint8_t* mb = r->mailbox + (uint64_t)index * r->mailbox_stride;
    // Resolve rids first: an unregistered hash must not seal the mailbox.
    Ingress* g = w->ingress.get();
    if (!resolve_rids(r->ledger, reqs, h.request_count, mb + kMailRids, &dropped,
                      g ? g->handles : nullptr, g ? g->gens : nullptr))
      return false;
    uint32_t token_total = 0;
    for (uint32_t i = 0; i < h.request_count; ++i) {
      uint32_t cnt;
      memcpy(&cnt, reqs + (uint64_t)i * 32 + 20, 4);  // token_count field
      token_total += cnt;
    }
    memcpy(mb + kMailMarkerB, slot + 3968, 8);
    memcpy(mb + kMailEvent, slot + kRecordV2EventHandleOffset, 64);
    *reinterpret_cast<uint64_t*>(mb + kMailDropped) = dropped;
    mailbox_publish(r, index, 2, h.request_count, token_total, slot,
                    sizeof(h), reqs, h.request_count * 32, tokens,
                    token_total * 4, nullptr, 0);
    stamp(w, dt_probe::MAILBOX_PUBLISHED, r, r->next_seq, h.request_count);
    if (dropped == (h.request_count >= 64 ? ~0ull : ((1ull << h.request_count) - 1))) {
      // Every request is released: no landing copy is needed — the draft
      // will never read this wave, so the bare INPUT_READY is the true
      // read proof, and the ring moves on instead of stalling.
      return !w->ingress || publish_v2(w, r, h, index, dropped, -1, 0, served_ns);
    }
  }
  if (!enqueue_pull(w, r, index, h.source_byte_offset, h.nbytes,
                    slot + kRecordV2EventHandleOffset))
    return false;
  return !w->ingress || publish_v2(w, r, h, index, dropped, r->placement_buffer[index],
                                   r->placement_offset[index], served_ns);
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
  const uint64_t served_ns = w->ingress ? now_ns() : 0;
  r->placement_buffer[index] = -1;
  r->placement_offset[index] = 0;
  if (!mailbox_free(w, r)) return false;
  if (r->mailbox != nullptr) {
    if (h.payload_len > r->entry_bytes - 128) {
      fail(w, -2, r->id);
      return false;
    }
    mailbox_publish(r, index, 1, h.payload_len, 0, slot, sizeof(h), nullptr, 0,
                    nullptr, 0, slot + 128, h.payload_len);
    stamp(w, dt_probe::MAILBOX_PUBLISHED, r, r->next_seq);
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
  if (w->ingress) {
    const int k = r->placement_buffer[index];
    if (!publish_v1(w, r, r->next_seq, index, k, k >= 0 ? w->landing_gen[k] : 0,
                    r->placement_offset[index], served_ns))
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
  if (w->probe != nullptr && r->seen_seq != r->next_seq) {
    r->seen_seq = r->next_seq;
    stamp(w, dt_probe::RECORD_SEEN, r, r->next_seq);
  }
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

// Fences of one arrived cross-host entry: the local v1 fences (seq and epoch
// lap, format, flags, payload length) plus the registered-source fences the
// worker applies to kernel-published records (the slot's generation, here as
// mirrored onto the ring by the Target, and a row-aligned range inside that
// slot that fits one landing buffer).  The arrival completion already proved
// the record complete, so a seq mismatch is a stale or forged record, never
// "not yet published".  0: valid; -4: malformed record; -5: stale source.
int remote_record_check(const EntryHeader& h, uint64_t expected_seq, uint64_t payload_cap,
                        const uint32_t* table, uint64_t slot_count,
                        uint64_t allocation_offset, uint64_t slot_nbytes,
                        uint64_t row_bytes, uint64_t landing_bytes) {
  if (h.seq != (uint32_t)expected_seq || h.epoch != (uint32_t)(expected_seq >> 32)) return -4;
  if ((uint32_t)(h.source_offset >> 32) == kRecordVersion2) return -4;
  if (h.flags & ~kEntryHasPull || h.payload_len > payload_cap) return -4;
  if (!(h.flags & kEntryHasPull))
    return h.nbytes || h.source_slot_id || h.slot_generation || h.source_offset ? -4 : 0;
  if (table == nullptr || h.nbytes == 0 || h.nbytes % row_bytes || h.nbytes > landing_bytes)
    return -4;
  if (h.source_slot_id >= slot_count || h.slot_generation == 0 ||
      table[h.source_slot_id] != h.slot_generation)
    return -5;
  const uint64_t slot_base = allocation_offset + (uint64_t)h.source_slot_id * slot_nbytes;
  if (h.source_offset < slot_base || (h.source_offset - slot_base) % row_bytes ||
      h.source_offset + h.nbytes > slot_base + slot_nbytes)
    return -5;
  return 0;
}

// Consume every arrived ready record: validated exactly as
// dt_rdma.decode_ready_record does, then each receive is re-posted.
bool drain_ready(Worker* w, Ring* r) {
  Remote& m = *r->remote;
  if (!m.spec.ready_qp) return true;
  uint64_t wr_id[64];
  uint32_t status[64], opcode[64], imm[64], qpn[64];
  const int n = m.poll(reinterpret_cast<void*>(m.spec.ready_recv_cq), 64, 0, wr_id, status,
                       opcode, imm, qpn);
  if (n < 0) {
    fail(w, -7, r->id);
    return false;
  }
  for (int i = 0; i < n; ++i) {
    if (status[i] != m.spec.wc_success || opcode[i] != m.spec.wc_recv_with_imm) {
      fail(w, -7, r->id);
      return false;
    }
    const RemoteRingSpec& s = m.spec;
    const uint8_t* rec = reinterpret_cast<const uint8_t*>(s.ready_ring_host) +
                         (uint64_t)(imm[i] % s.ready_depth) * s.ready_record_bytes;
    const uint64_t slot_id = ready_field(rec, s.ready_slot_id_at, s.ready_slot_id_bytes);
    const uint64_t generation =
        ready_field(rec, s.ready_slot_generation_at, s.ready_slot_generation_bytes);
    if (ready_field(rec, s.ready_session_tag_at, s.ready_session_tag_bytes) != s.session_tag ||
        ready_field(rec, s.ready_pool_tag_at, s.ready_pool_tag_bytes) != s.pool_tag ||
        ready_field(rec, s.ready_seq_at, s.ready_seq_bytes) != m.ready_expected ||
        slot_id >= s.slot_count || generation == 0) {
      fail(w, -6, r->id);
      return false;
    }
    ++m.ready_expected;
    if (generation > m.ready_table[slot_id])
      __atomic_store_n(&m.ready_table[slot_id], (uint32_t)generation, __ATOMIC_RELEASE);
  }
  if (n > 0 && m.post_recv(reinterpret_cast<void*>(m.spec.ready_qp), m.spec.ready_tag, n) != 0) {
    fail(w, -7, r->id);
    return false;
  }
  return true;
}

// Serve the oldest arrived record: 1 served, 0 retry next pass, -1 failed.
int serve_remote_record(Worker* w, Ring* r) {
  Remote& m = *r->remote;
  const uint32_t index = r->next_seq % r->capacity;
  if (m.arrived.front() != index) {
    fail(w, -4, r->id);
    return -1;
  }
  const uint8_t* slot = r->entries + (uint64_t)index * r->entry_bytes;
  EntryHeader h;
  memcpy(&h, slot, sizeof(h));
  const int check = remote_record_check(
      h, r->next_seq, r->entry_bytes - 128, m.spec.ready_qp ? m.table : nullptr,
      m.spec.slot_count, m.spec.pool_allocation_offset, m.spec.slot_nbytes, w->row_bytes,
      w->landing_bytes);
  if (check != 0) {
    fail(w, check, r->id);
    return -1;
  }
  const bool has_pull = h.flags & kEntryHasPull;
  if (has_pull) {
    // The Target's producer has not finished this slot generation yet.
    const uint32_t ready =
        __atomic_load_n(&m.ready_table[h.source_slot_id], __ATOMIC_ACQUIRE);
    if (ready < h.slot_generation) return 0;
    if (ready != h.slot_generation) {
      fail(w, -5, r->id);  // republished while this record still leases it
      return -1;
    }
    if (m.reads_outstanding >= m.spec.read_window) return 0;
  }
  if (!mailbox_free(w, r)) return 0;
  uint64_t offset = 0;
  int k = -1;
  if (has_pull) {
    k = place_locked(w, h.nbytes, &offset);
    if (k < 0) {
      w->landing_stalls.fetch_add(1, std::memory_order_relaxed);
      return 0;
    }
    stamp(w, dt_probe::PLACEMENT, r, r->next_seq, placement_arg(k, offset));
  }
  // Snapshot first: after the receive is re-posted the Target may rewrite
  // this ring entry.
  mailbox_publish(r, index, 1, h.payload_len, 0, slot, sizeof(h), nullptr, 0, nullptr, 0,
                  slot + 128, h.payload_len);
  stamp(w, dt_probe::MAILBOX_PUBLISHED, r, r->next_seq);
  if (!r->mail_pin.empty()) r->mail_pin[index] = UINT64_MAX;  // until its READY entry
  m.arrived.pop_front();
  if (m.post_recv(reinterpret_cast<void*>(m.spec.record_qp), m.spec.record_tag, 1) != 0) {
    fail(w, -7, r->id);
    return -1;
  }
  r->placement_buffer[index] = k;
  r->placement_offset[index] = offset;
  if (has_pull) {
    if (timed(w->probe, dt_probe::RDMA_POST_READ, [&] {
          return m.post_read(reinterpret_cast<void*>(m.spec.record_qp), m.spec.record_tag,
                             w->landing_base + (uint64_t)k * w->landing_bytes + offset,
                             (uint32_t)h.nbytes, (uint32_t)m.spec.landing_lkey,
                             m.spec.pool_addr + h.source_offset - m.spec.pool_allocation_offset,
                             (uint32_t)m.spec.pool_rkey);
        }) != 0) {
      fail(w, -7, r->id);
      return -1;
    }
    stamp(w, dt_probe::READ_POSTED, r, r->next_seq, h.nbytes);
    w->cursor = offset + h.nbytes;
    w->landing_placed[k] += 1;
    ++m.reads_outstanding;
  }
  m.flights.push_back(Flight{r->next_seq, index, h.source_slot_id, h.slot_generation, has_pull,
                             !has_pull, k >= 0 ? w->landing_gen[k] : 0,
                             w->ingress ? now_ns() : 0});
  advance_seq(r);
  return 1;
}

// Publish INPUT_READY for the completed prefix of served entries, with one
// GPU-visibility fence for every READ among them.
bool publish_remote(Worker* w, Ring* r) {
  Remote& m = *r->remote;
  size_t count = 0;
  bool any_read = false;
  for (const Flight& f : m.flights) {
    if (!f.done) break;
    ++count;
    any_read |= f.read;
  }
  if (!count) return true;
  if (any_read) {
    // A republication whose ready record arrived meanwhile may have
    // overwritten a range under its READ.
    if (!drain_ready(w, r)) return false;
    for (size_t i = 0; i < count; ++i) {
      const Flight& f = m.flights[i];
      if (f.read &&
          __atomic_load_n(&m.ready_table[f.slot_id], __ATOMIC_ACQUIRE) != f.slot_generation) {
        fail(w, -5, r->id);
        return false;
      }
    }
    if (CUresult rc = timed(w->probe, dt_probe::FLUSH_RDMA_WRITES, [&] {
          return w->api.cuFlushGPUDirectRDMAWrites((int)m.spec.flush_target,
                                                   (int)m.spec.flush_scope);
        });
        rc != 0) {
      fail(w, rc, r->id);
      return false;
    }
  }
  for (size_t i = 0; i < count; ++i) {
    const Flight& f = m.flights.front();
    // READY after the READ completed and the flush above (§6.5, cross host).
    if (w->ingress && !publish_v1(w, r, f.seq, f.index, r->placement_buffer[f.index],
                                  f.landing_gen, r->placement_offset[f.index], f.served_ns))
      return false;
    __atomic_store_n(&m.ready_host[f.index], (uint32_t)f.seq, __ATOMIC_RELEASE);
    r->consumed_seq.store(f.seq, std::memory_order_release);
    stamp(w, dt_probe::INPUT_READY_ISSUED, r, f.seq);
    m.flights.pop_front();
    w->entries_done.fetch_add(1, std::memory_order_relaxed);
  }
  return true;
}

// Reap READ completions; they complete in posting order on one RC queue pair.
bool reap_reads(Worker* w, Ring* r) {
  Remote& m = *r->remote;
  if (!m.reads_outstanding) return true;
  uint64_t wr_id[64];
  uint32_t status[64], opcode[64], imm[64], qpn[64];
  const int n = m.poll(reinterpret_cast<void*>(m.spec.record_send_cq), 64, 0, wr_id, status,
                       opcode, imm, qpn);
  if (n < 0) {
    fail(w, -7, r->id);
    return false;
  }
  m.reads_outstanding -= n;  // error completions retire their READs too
  for (int i = 0; i < n; ++i) {
    if (status[i] != m.spec.wc_success) {
      fail(w, -7, r->id);  // a failed READ (or a QP in error) is fail-stop
      return false;
    }
    for (Flight& f : m.flights)
      if (f.read && !f.done) {
        f.done = true;
        stamp(w, dt_probe::READ_COMPLETED, r, f.seq);
        break;
      }
  }
  return true;
}

// One pass over a cross-host ring; true when anything moved.
bool serve_remote(Worker* w, Ring* r) {
  Remote& m = *r->remote;
  const uint64_t done_before = w->entries_done.load(std::memory_order_relaxed);
  if (!drain_ready(w, r) || !reap_reads(w, r)) return false;
  uint64_t wr_id[64];
  uint32_t status[64], opcode[64], imm[64], qpn[64];
  const int n = m.poll(reinterpret_cast<void*>(m.spec.record_recv_cq), 64, 0, wr_id, status,
                       opcode, imm, qpn);
  if (n < 0) {
    fail(w, -7, r->id);
    return false;
  }
  for (int i = 0; i < n; ++i) {
    if (status[i] != m.spec.wc_success || opcode[i] != m.spec.wc_recv_with_imm) {
      fail(w, -7, r->id);
      return false;
    }
    stamp(w, dt_probe::RECORD_SEEN, r, r->next_seq + m.arrived.size());
    m.arrived.push_back(imm[i] % r->capacity);
  }
  bool served = false;
  for (int i = 0; i < 64 && !m.arrived.empty(); ++i) {
    const int rc = serve_remote_record(w, r);
    if (rc < 0) return false;
    if (rc == 0) break;
    served = true;
  }
  if (!publish_remote(w, r)) return false;
  return n > 0 || served || w->entries_done.load(std::memory_order_relaxed) != done_before;
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
      if (w->ingress) ingress_reclaim(w);
      // Register for the whole pass so writer-side reclamation can only fire
      // between passes, when this thread provably holds no entry.  All rings
      // share one ledger in production; opening per ring stays correct if
      // that ever changes.
      for (Ring* r : w->rings) {
        if (r->detached) continue;
        if (r->remote) {
          if (serve_remote(w, r)) progressed = true;
          continue;
        }
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
    uint32_t gen;   // per cell: +1 per registration; a tombstone keeps its request's
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
  const RidLedger::Entry* prev =
      l->slots[first_dead].load(std::memory_order_acquire);
  RidLedger::Entry* body = new RidLedger::Entry();
  body->key = key;
  memcpy(body->rid, rid, strlen(rid) + 1);
  body->released = false;
  body->gen = prev != nullptr ? prev->gen + 1 : 1;
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
// -5: released hash (tombstone; the record must be consumed and dropped); its
// rid is still copied.  `cell`/`gen` (the request handle, D6) are set for a
// live or released hash.  The guard registers this thread for the probe's
// duration, so the writer's reclamation can never free an entry mid-read.
int dt_rid_ledger_lookup(void* handle, uint64_t hash, char* out, uint32_t out_len,
                         uint32_t* cell, uint32_t* gen) {
  RidLedger* l = static_cast<RidLedger*>(handle);
  if (!l || !out || !out_len) return -3;
  LedgerReadGuard guard(l);
  const uint64_t key = hash + 1;
  const uint64_t mask = l->slots.size() - 1;
  for (size_t i = key & mask;; i = (i + 1) & mask) {
    const RidLedger::Entry* cur = l->slots[i].load(std::memory_order_acquire);
    if (cur == nullptr) return 0;
    if (cur->key == key) {
      const size_t len = strlen(cur->rid);
      if (len + 1 > out_len) return -4;
      memcpy(out, cur->rid, len + 1);
      *cell = (uint32_t)i;
      *gen = cur->gen;
      return cur->released ? -5 : (int)len;
    }
    if (i == ((key - 1) & mask)) return 0;
  }
}

int dt_rid_ledger_resolve(void* handle, uint64_t hash, char* out, uint32_t out_len) {
  uint32_t cell, gen;
  return dt_rid_ledger_lookup(handle, hash, out, out_len, &cell, &gen);
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
  w->landing_released.assign(landing_count, 0);
  w->landing_gen.assign(landing_count, 0);
  int least = 0, greatest = 0;
  w->api.cuCtxSetCurrent(w->ctx);
  w->api.cuCtxGetStreamPriorityRange(&least, &greatest);
  w->stream_priority = greatest;
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
                          void* ledger, const char* target_id, uint64_t dest_host,
                          uint64_t dest_stride, uint64_t dest_seq_at) {
  Worker* w = static_cast<Worker*>(handle);
  if (!w || !entries_host || entry_bytes < 128 || !capacity || !first_seq || !placement_buffer ||
      !placement_offset)
    return -1;
  if (mailbox_base &&
      (mailbox_stride < kMailMinStride || !header_host || (!mail_ack_host && !w->ingress)))
    return -1;
  // Native ingress parses from the mailbox and hashes with the Target id.
  if (w->ingress && (!mailbox_base || !target_id || strlen(target_id) >= sizeof(Ring::target_id) ||
                     (dest_host && dest_seq_at + 8 > dest_stride)))
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
  if (w->ingress) {
    memcpy(r->target_id, target_id, strlen(target_id) + 1);
    r->dest = reinterpret_cast<const uint8_t*>(dest_host);
    r->dest_stride = dest_stride;
    r->dest_seq_at = dest_seq_at;
    r->mail_pin.assign(capacity, 0);
  }
  w->api.cuCtxSetCurrent(w->ctx);
  // CU_STREAM_NON_BLOCKING: never serialize with the legacy default stream.
  if (w->api.cuStreamCreateWithPriority(&r->stream, 0x1, w->stream_priority) != 0) {
    delete r;
    return -1;
  }
  std::lock_guard<std::mutex> guard(w->lock);
  w->rings.push_back(r);
  return 0;
}

// Attach a cross-host ring (issue #198).  `entries_host` is entry 0 of the
// pinned ring the Target writes; `ready_words_host` the INPUT_READY words the
// main loop reads; the mailbox contract is the local one (v1 snapshots).
int dt_ring_worker_attach_remote(void* handle, int ring_id, uint64_t entries_host,
                                 uint64_t entry_bytes, uint32_t capacity,
                                 uint64_t ready_words_host, uint64_t placement_buffer,
                                 uint64_t placement_offset, uint64_t first_seq,
                                 uint64_t mailbox_base, uint64_t mailbox_stride,
                                 uint64_t mail_ack_host, const RemoteRingSpec* spec) {
  Worker* w = static_cast<Worker*>(handle);
  if (!w || !spec || !entries_host || entry_bytes < 128 || !capacity || !first_seq ||
      !ready_words_host || !placement_buffer || !placement_offset || !mailbox_base ||
      mailbox_stride < kMailMinStride || (!mail_ack_host && !w->ingress) || !spec->post_read ||
      !spec->post_recv || !spec->poll || !spec->record_qp || !spec->record_recv_cq ||
      !spec->record_send_cq || !spec->read_window ||
      (spec->ready_qp && (!spec->ready_recv_cq || !spec->ready_ring_host ||
                          !spec->ready_depth || !spec->ready_table_host ||
                          !spec->table_host || !spec->slot_nbytes ||
                          !ready_layout_ok(*spec))))
    return -1;
  Ring* r = new Ring();
  r->id = ring_id;
  r->entries = reinterpret_cast<const uint8_t*>(entries_host);
  r->entry_bytes = entry_bytes;
  r->capacity = capacity;
  r->placement_buffer = reinterpret_cast<int32_t*>(placement_buffer);
  r->placement_offset = reinterpret_cast<uint64_t*>(placement_offset);
  r->next_seq = first_seq;
  r->mailbox = reinterpret_cast<uint8_t*>(mailbox_base);
  r->mailbox_stride = mailbox_stride;
  r->mail_ack = reinterpret_cast<const uint64_t*>(mail_ack_host);
  if (w->ingress) r->mail_pin.assign(capacity, 0);
  r->remote.reset(new Remote());
  Remote& m = *r->remote;
  m.spec = *spec;
  m.post_read = reinterpret_cast<PostReadFn>(spec->post_read);
  m.post_recv = reinterpret_cast<PostRecvFn>(spec->post_recv);
  m.poll = reinterpret_cast<PollFn>(spec->poll);
  m.ready_host = reinterpret_cast<uint32_t*>(ready_words_host);
  m.table = reinterpret_cast<const uint32_t*>(spec->table_host);
  m.ready_table = reinterpret_cast<uint32_t*>(spec->ready_table_host);
  m.ready_expected = spec->ready_first_seq;
  std::lock_guard<std::mutex> guard(w->lock);
  w->rings.push_back(r);
  return 0;
}

// CPU-testable form of the cross-host record fences (see remote_record_check).
int dt_ring_remote_record_check(const uint8_t* entry, uint64_t entry_bytes,
                                uint64_t expected_seq, const uint32_t* table,
                                uint64_t slot_count, uint64_t allocation_offset,
                                uint64_t slot_nbytes, uint64_t row_bytes,
                                uint64_t landing_bytes) {
  EntryHeader h;
  memcpy(&h, entry, sizeof(h));
  return remote_record_check(h, expected_seq, entry_bytes - 128, table, slot_count,
                             allocation_offset, slot_nbytes, row_bytes, landing_bytes);
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
  const uint64_t placed = w->landing_placed[k];
  if (w->ingress) landing_free_if_released(w, k);
  return placed;
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

// Turn on native ingress (D6); only before start and before any attach.
// The READY queue and the token arena hold twice the resident landing rows:
// every READY entry of a pulled entry names at least one landing row, so
// neither fills before the landing buffers do.  `wake_set` (dt_wake_set) and
// `wake` may be null; otherwise `wake_bit` is set after every publish.
int dt_ring_worker_enable_ingress(void* handle, void (*wake_set)(void*, uint32_t), void* wake,
                                  uint32_t wake_bit) {
  Worker* w = static_cast<Worker*>(handle);
  if (w->thread.joinable() || w->ingress || !w->rings.empty()) return -1;
  const uint64_t rows = w->landing_free.size() * (w->landing_bytes / w->row_bytes);
  uint64_t capacity = 1;
  while (capacity < 2 * rows) capacity <<= 1;
  auto g = std::make_unique<Ingress>();
  g->queue.resize(capacity);
  g->tokens.resize(2 * rows + 512);  // + one record's tokens of wrap padding
  g->releases.resize(capacity);
  g->wake_set = wake_set;
  g->wake = wake;
  g->wake_bit = wake_bit;
  w->ingress = std::move(g);
  return 0;
}

// Scheduler side of the READY queue: copy up to `max` entries starting at
// READY seq `from` into `out`; returns the count, or -1 when an entry's seq
// is not the one expected there (R7).  Nothing is reclaimed until
// dt_ring_worker_ready_consume passes it.
int64_t dt_ring_worker_ready_drain(void* handle, uint64_t from, ReadyEntry* out, uint64_t max) {
  Ingress* g = static_cast<Worker*>(handle)->ingress.get();
  if (!g || !from) return -1;
  const uint64_t published = g->published.load(std::memory_order_acquire);
  uint64_t n = 0;
  for (uint64_t seq = from; seq <= published && n < max; ++seq, ++n) {
    out[n] = g->queue[(seq - 1) % g->queue.size()];
    if (out[n].seq != seq) return -1;
  }
  return (int64_t)n;
}

// ready_consumed_seq: the scheduler is done with every entry up to `seq`
// (its copy event waited on, its token ids and v1 payload read).
void dt_ring_worker_ready_consume(void* handle, uint64_t seq) {
  static_cast<Worker*>(handle)->ingress->consumed.store(seq, std::memory_order_release);
}

// Producer side of the landing release queue (one producer thread): `count`
// placements of buffer `k` at generation `gen` are no longer read.  0: queued;
// -1: no native ingress; -2: queue full.
int dt_ring_worker_landing_release(void* handle, uint32_t k, uint32_t gen, uint64_t count) {
  Ingress* g = static_cast<Worker*>(handle)->ingress.get();
  if (!g) return -1;
  const uint64_t head = g->release_head.load(std::memory_order_relaxed);
  if (head - g->release_tail.load(std::memory_order_acquire) >= g->releases.size()) return -2;
  g->releases[head % g->releases.size()] = Ingress::Release{k, gen, count};
  g->release_head.store(head + 1, std::memory_order_release);
  return 0;
}

// Address of a ring's consumed seq (a u64 the worker stores with release);
// valid until destroy.  0: no such ring.
uint64_t dt_ring_worker_consumed_seq_address(void* handle, int ring_id) {
  Worker* w = static_cast<Worker*>(handle);
  std::lock_guard<std::mutex> guard(w->lock);
  for (Ring* r : w->rings)
    if (r->id == ring_id) return reinterpret_cast<uint64_t>(&r->consumed_seq);
  return 0;
}

// The READY entry layout, field by field (nullptr past the last field), so
// the Python mirror never copies it.
const char* dt_ring_ready_field(uint32_t index, uint64_t* offset, uint64_t* size,
                                int* is_signed) {
#define DT_READY_FIELD(name)                                            \
  {#name, offsetof(ReadyEntry, name), sizeof(ReadyEntry::name),          \
   std::is_signed<decltype(ReadyEntry::name)>::value},
  struct Field {
    const char* name;
    uint64_t offset, size;
    bool is_signed;
  };
  static const Field fields[] = {DT_READY_FIELDS(DT_READY_FIELD)};
#undef DT_READY_FIELD
  if (index >= sizeof(fields) / sizeof(fields[0])) return nullptr;
  *offset = fields[index].offset;
  *size = fields[index].size;
  *is_signed = fields[index].is_signed;
  return fields[index].name;
}

// The READY kind and sentinel values by index (nullptr past the last).
const char* dt_ring_ready_constant(uint32_t index, uint64_t* value) {
#define DT_READY_CONSTANT(name, v) {#name, v},
  struct Constant {
    const char* name;
    uint64_t value;
  };
  static const Constant constants[] = {DT_READY_CONSTANTS(DT_READY_CONSTANT)};
#undef DT_READY_CONSTANT
  if (index >= sizeof(constants) / sizeof(constants[0])) return nullptr;
  *value = constants[index].value;
  return constants[index].name;
}

// CPU-testable native v2 parse (no CUDA): the worker's rid resolution and
// parse_v2_rows over one kernel-published record image at seq `seq`, as if
// placed at landing row `landing_row0` of buffer 0.  `dest` is the record's
// destination-region slot or null.  Returns the entry count (token ids in
// `tokens`, token_at relative to it), 0 for an unregistered rid, <0 on error.
int dt_ring_parse_v2_record(const uint8_t* record, void* ledger, const char* target_id,
                            const uint8_t* dest, uint64_t dest_seq_at, int ring_id,
                            uint64_t landing_row0, ReadyEntry* out, uint32_t* tokens) {
  RecordV2Header h;
  memcpy(&h, record, sizeof(h));
  if (h.request_count > kRecordV2MaxReq) return -4;
  std::vector<uint8_t> mb(kMailMinStride);
  memcpy(mb.data() + kMailReqs, record + 128, (uint64_t)h.request_count * 32);
  memcpy(mb.data() + kMailTokens, record + 128 + 56 * 32, 512 * 4);
  uint64_t dropped = 0;
  uint32_t handles[kRecordV2MaxReq], gens[kRecordV2MaxReq], ntok = 0;
  if (!resolve_rids(ledger, record + 128, h.request_count, mb.data() + kMailRids, &dropped,
                    handles, gens))
    return 0;
  Ring r;
  r.id = ring_id;
  ReadyEntry base = ready_base(&r, h.seq, 0);
  base.landing_k = 0;
  return parse_v2_rows(h, mb.data(), dropped, handles, gens, target_id, dest, dest_seq_at, base,
                       landing_row0, out, tokens, &ntok);
}

// Attach a dt_probe_create probe (owned by the caller, outliving the
// worker); only before start, so the thread reads a fixed pointer.
int dt_ring_worker_set_probe(void* handle, void* probe) {
  Worker* w = static_cast<Worker*>(handle);
  if (w->thread.joinable()) return -1;
  w->probe = static_cast<dt_probe::Probe*>(probe);
  return 0;
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
  int rc = w->api.cuStreamSynchronize(w->stream);
  std::vector<CUstream> ring_streams;
  {
    std::lock_guard<std::mutex> guard(w->lock);
    ring_streams.reserve(w->rings.size());
    for (Ring* r : w->rings) ring_streams.push_back(r->stream);
  }
  for (CUstream s : ring_streams)
    if (s != nullptr)  // cross-host rings have no stream
      if (int rrc = w->api.cuStreamSynchronize(s); rrc != 0 && rc == 0) rc = rrc;
  return rc;
}

// Stop the thread, drain every stream, release everything it owns.
int dt_ring_worker_destroy(void* handle) {
  Worker* w = static_cast<Worker*>(handle);
  if (!w) return 0;
  w->stop.store(true);
  if (w->thread.joinable()) w->thread.join();
  w->api.cuCtxSetCurrent(w->ctx);
  int rc = w->api.cuStreamSynchronize(w->stream);
  for (Ring* r : w->rings) {
    // Posted READs still write the landing buffers: reap them before the
    // caller may release those buffers.  A READ that never completes leaves
    // the QP to the owner's close (destroying it stops the NIC).
    if (r->remote) {
      const uint64_t deadline = now_ns() + r->remote->spec.drain_timeout_ns;
      while (r->remote->reads_outstanding && now_ns() < deadline) {
        const uint64_t before = r->remote->reads_outstanding;
        if (!reap_reads(w, r)) break;
        if (r->remote->reads_outstanding == before) sched_yield();
      }
      if (r->remote->reads_outstanding && rc == 0) rc = -7;
    }
    // cuStreamDestroy returns immediately when the stream has pending work and
    // releases it lazily; drain first so no copy or ready write can still touch
    // buffers the caller is about to release after we return.
    if (r->stream) {
      if (int rrc = w->api.cuStreamSynchronize(r->stream); rrc != 0 && rc == 0) rc = rrc;
      w->api.cuStreamDestroy(r->stream);
    }
    delete r;
  }
  if (w->ingress)  // every ring stream drained: no record is pending
    for (CUevent event : w->ingress->events) w->api.cuEventDestroy(event);
  w->api.cuStreamDestroy(w->stream);
  w->api.cuDevicePrimaryCtxRelease(w->device);
  delete w;
  return rc;
}

}  // extern "C"
