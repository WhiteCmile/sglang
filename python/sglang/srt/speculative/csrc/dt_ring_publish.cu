// DT V2 §7.1 GPU ring publish kernel (DT_RING_PUBLISH=1).
//
// One non-resident 1-block kernel fills a full 4096-byte ring record for one
// verify wave and publishes the seq word last with system-scope release
// semantics, so the Draft's GIL-free ring worker (dt_ring_worker.cpp) polling
// with acquire sees a fully consistent record.  The kernel runs on the
// Target's verify stream immediately after the feature pack copy and the
// producer-event record, replacing the Python publication loop on the
// scheduler thread.
//
// Record layout (mirrors RING_PUBLISH_* constants in dt_direct_ring.py and
// the RecordV2 parsing in dt_ring_worker.cpp):
//
//   Header (128 B):
//     0   u64 seq                 published LAST, st.release.sys
//     8   u32 flags               bit0: HAS_PULL
//     12  u32 version             = 2 (old CPU entries keep 0 here)
//     16  u64 owner_epoch         Coordinator owner-lease epoch (from staging)
//     24  u32 source_slot_id      launch arg
//     28  u32 slot_generation     launch arg
//     32  u64 source_byte_offset  slot base offset in the registered source
//     40  u64 nbytes              total hidden bytes (row_count * row_bytes)
//     48  u32 row_count
//     52  u32 row_bytes
//     56  u32 request_count
//     60  u32 block_size
//     64  u64 hidden_produced_ns  %globaltimer at publish
//     72  u64 marker_a            ~seq
//     80  48 B reserved
//   128   56 x 32 B request table:
//     +0  u64 rid_hash    +8  u32 round_id   +12 u32 attempt
//     +16 u32 token_offset (== row offset into the slot's rows)
//     +20 u32 token_count (committed rows: accept count + bonus)
//     +24 u32 bonus_token +28 u32 seq_len
//   1920  512 x u32 token ids (aligned 1:1 with feature rows)
//   3968  u64 marker_b            whole-record visibility proof
//   3976  64 B producer-event IPC handle, written by the CPU BEFORE launch
//         (validated: the kernel's system-scope release publishes the prior
//         CPU store, see the ring-publish prototype cpu-field experiment)
//   4040  56 B pad
//
// Staging buffer (host-pinned, device-mapped, written by the scheduler with
// plain stores before launch; one 32 B entry per batch slot):
//   +0  u64 rid_hash   +8 u32 round_id   +12 u32 attempt
//   +16 u64 owner_epoch
//   +24 u64 token_src  device pointer to the request's committed candidate
//                      tokens (int64, token_count contiguous values)
// The active batch slots are listed compactly in a u32 array at
// STAGING_ACTIVE_OFFSET; entry i of the request table uses slot active[i].

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr int ENTRY_BYTES = 4096;
constexpr int HEADER_BYTES = 128;
constexpr int MAX_REQ = 56;
constexpr int REQ_ENTRY_BYTES = 32;
constexpr int MAX_TOK = 512;
constexpr int TOKEN_OFFSET = HEADER_BYTES + MAX_REQ * REQ_ENTRY_BYTES;  // 1920
constexpr int TAIL_OFFSET = TOKEN_OFFSET + MAX_TOK * 4;                 // 3968
constexpr uint32_t VERSION = 2;
constexpr uint32_t FLAG_HAS_PULL = 1;
constexpr uint64_t MARKER_B_XOR = 0x9E3779B97F4A7C15ull;

// Staging layout (must match dt_ring_publish_ext.py / dt_direct_ring.py):
// per-slot 32 B entries start at `staging_entries_offset` (launch arg); the
// compact u32 active-slot list sits at `staging_active_offset` (launch arg).

struct Header {
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
static_assert(sizeof(Header) == HEADER_BYTES, "header layout");

struct ReqEntry {
  uint64_t rid_hash;
  uint32_t round_id;
  uint32_t attempt;
  uint32_t token_offset;
  uint32_t token_count;
  uint32_t bonus_token;
  uint32_t seq_len;
};
static_assert(sizeof(ReqEntry) == REQ_ENTRY_BYTES, "req layout");

constexpr int STAGING_ENTRY_BYTES = 32;

struct StagingEntry {
  uint64_t rid_hash;
  uint32_t round_id;
  uint32_t attempt;
  uint64_t owner_epoch;
  uint64_t token_src;
};
static_assert(sizeof(StagingEntry) == STAGING_ENTRY_BYTES, "staging layout");

__device__ __forceinline__ uint64_t gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void st_rel_sys64(uint64_t* p, uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

// Non-resident: one block fills one record, publishes seq last, exits.
//
// ring_dev:      device pointer to the ring entries region (host-mapped).
// staging_dev:   device pointer to the staging buffer (host-mapped).
// counts_dev:    device pointer to per-batch-slot committed token counts
//                (int32, indexed by batch slot, host-mapped like staging).
// bonus_dev:     device pointer to per-batch-slot bonus tokens (int64).
// seq_lens_dev:  device pointer to per-batch-slot new seq lens (int64).
// offset_stride: 0 = feature row offsets are prefix sums of the committed
// counts (park-time packed layout); >0 = feature row offsets are
// batch_slot*offset_stride (in-place reservation layout, where the slot
// holds the full packed wave).  Token ids are always written at compacted
// prefix positions in the 512-entry token array; the record's token_offset
// field is the feature row offset, which the Draft maps back to a compacted
// token position by prefix-summing the request table.
__global__ void publish_k(uint8_t* ring_dev, uint64_t capacity, uint64_t seq,
                          uint32_t source_slot_id, uint32_t slot_generation,
                          uint64_t source_byte_offset, uint64_t nbytes,
                          uint32_t row_count, uint32_t row_bytes,
                          uint32_t request_count, uint32_t block_size,
                          uint32_t offset_stride,
                          uint8_t* staging_dev, uint64_t staging_entries_offset,
                          uint64_t staging_active_offset,
                          const int32_t* counts_dev, const int64_t* bonus_dev,
                          const int64_t* seq_lens_dev) {
  uint8_t* slot = ring_dev + (seq % capacity) * (uint64_t)ENTRY_BYTES;
  Header* h = (Header*)slot;
  ReqEntry* reqs = (ReqEntry*)(slot + HEADER_BYTES);
  uint32_t* toks = (uint32_t*)(slot + TOKEN_OFFSET);
  uint64_t* tail = (uint64_t*)(slot + TAIL_OFFSET);

  // Staging is host memory written by the scheduler before launch; volatile
  // loads keep the compiler from caching stale values.  Entries are compacted
  // in active order: entry i describes active[i] (the active list is needed
  // only to index the per-batch-slot GPU tensors).
  const volatile uint32_t* active =
      (const volatile uint32_t*)(staging_dev + staging_active_offset);
  const volatile StagingEntry* staging =
      (const volatile StagingEntry*)(staging_dev + staging_entries_offset);

  __shared__ uint32_t offsets[MAX_REQ + 1];
  __shared__ uint32_t tokpos[MAX_REQ + 1];
  __shared__ uint64_t owner_epoch_sh;
  if (threadIdx.x == 0) {
    uint32_t prefix = 0;
    for (uint32_t i = 0; i < request_count; ++i) {
      uint32_t batch_slot = active[i];
      tokpos[i] = prefix;
      offsets[i] = offset_stride ? batch_slot * offset_stride : prefix;
      prefix += (uint32_t)counts_dev[batch_slot];
    }
    offsets[request_count] = prefix;
    owner_epoch_sh = request_count ? staging[0].owner_epoch : 0;
  }
  __syncthreads();

  // One thread per request fills its table entry and its token range.
  for (uint32_t i = threadIdx.x; i < request_count; i += blockDim.x) {
    uint32_t batch_slot = active[i];
    StagingEntry s;
    const volatile StagingEntry* vs = &staging[i];
    s.rid_hash = vs->rid_hash;
    s.round_id = vs->round_id;
    s.attempt = vs->attempt;
    s.owner_epoch = vs->owner_epoch;
    s.token_src = vs->token_src;
    uint32_t count = (uint32_t)counts_dev[batch_slot];
    uint32_t offset = offsets[i];
    uint32_t tpos = tokpos[i];
    ReqEntry e;
    e.rid_hash = s.rid_hash;
    e.round_id = s.round_id;
    e.attempt = s.attempt;
    e.token_offset = offset;
    e.token_count = count;
    e.bonus_token = (uint32_t)bonus_dev[batch_slot];
    e.seq_len = (uint32_t)seq_lens_dev[batch_slot];
    reqs[i] = e;
    const int64_t* src = (const int64_t*)s.token_src;
    for (uint32_t j = 0; j < count; ++j) toks[tpos + j] = (uint32_t)src[j];
  }
  if (threadIdx.x == 0) {
    h->flags = FLAG_HAS_PULL;
    h->version = VERSION;
    h->owner_epoch = owner_epoch_sh;
    h->source_slot_id = source_slot_id;
    h->slot_generation = slot_generation;
    h->source_byte_offset = source_byte_offset;
    h->nbytes = nbytes;
    h->row_count = row_count;
    h->row_bytes = row_bytes;
    h->request_count = request_count;
    h->block_size = block_size;
    h->marker_a = ~seq;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    *tail = seq ^ MARKER_B_XOR;
    h->hidden_produced_ns = gtime();
    // Publish seq last with system-scope release; this also covers the CPU's
    // pre-launch store of the producer-event handle at EVENT_HANDLE_OFFSET.
    __threadfence_system();
    st_rel_sys64(&h->seq, seq);
  }
}

}  // namespace

extern "C" {

int rp_init(int device) { return (int)cudaSetDevice(device); }

// Pinned device-mapped staging buffer for the scheduler's scalar stores.
int64_t rp_staging_alloc(int64_t bytes, int* rc) {
  void* p = nullptr;
  cudaError_t e = cudaHostAlloc(&p, bytes, cudaHostAllocMapped);
  *rc = (int)e;
  if (e == cudaSuccess) memset(p, 0, bytes);
  return (int64_t)p;
}

int64_t rp_dev_ptr(int64_t host, int* rc) {
  void* d = nullptr;
  cudaError_t e = cudaHostGetDevicePointer(&d, (void*)host, 0);
  *rc = (int)e;
  return (int64_t)d;
}

int rp_staging_free(int64_t host) { return (int)cudaFreeHost((void*)host); }

// All integer args are passed as int64 from ctypes and narrowed here.
int rp_publish(int64_t stream, int64_t ring_dev, int64_t capacity, int64_t seq,
               int64_t source_slot_id, int64_t slot_generation,
               int64_t source_byte_offset, int64_t nbytes, int64_t row_count,
               int64_t row_bytes, int64_t request_count, int64_t block_size,
               int64_t offset_stride, int64_t staging_dev,
               int64_t staging_entries_offset, int64_t staging_active_offset,
               int64_t counts_dev, int64_t bonus_dev, int64_t seq_lens_dev) {
  publish_k<<<1, 128, 0, (cudaStream_t)stream>>>(
      (uint8_t*)ring_dev, (uint64_t)capacity, (uint64_t)seq,
      (uint32_t)source_slot_id, (uint32_t)slot_generation,
      (uint64_t)source_byte_offset, (uint64_t)nbytes, (uint32_t)row_count,
      (uint32_t)row_bytes, (uint32_t)request_count, (uint32_t)block_size,
      (uint32_t)offset_stride, (uint8_t*)staging_dev,
      (uint64_t)staging_entries_offset, (uint64_t)staging_active_offset,
      (const int32_t*)counts_dev, (const int64_t*)bonus_dev,
      (const int64_t*)seq_lens_dev);
  return (int)cudaGetLastError();
}

}  // extern "C"
