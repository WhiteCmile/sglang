// Batched completion record publication for Draft->Target candidate pushes.
// Launched on the push stream after the candidate copy: the payload words are
// written first, then a system-scope release store publishes the reservation
// word last, so a Target CPU that observes the published word also observes
// every payload word.  The reservation word carries the reservation's
// authorization nonce (unique per reservation, known to both sides), so a
// stale record from a prior reservation or producer can never pass the
// Target's scan.
//
// The descriptor struct, kind values and record word indices come from
// dt_push_completion_layout.h, generated at library build time from the
// Python constants in dt_candidate_p2p.py (the single source of the ABI).
//
// Completion record layout per slot (64 bytes, naturally aligned):
//   0   u64 seq               (reservation authorization nonce, published LAST)
//   8   u32 generation
//   12  u32 valid_length
//   16  u64 transfer_id_hash
//   24  u32 root_token
//   28  u32 producer_tag
//   32  u64 produced_ns       (%globaltimer)
//   40  u64 globaltimer_offset_ns
//                             (producer-calibrated: realtime = globaltimer +
//                             offset; 0 means the producer is uncalibrated and
//                             the Target leaves delivery_ns at 0)
//   48  16B pad
#include <cstdint>
#include <cuda_runtime.h>

#include "dt_push_completion_layout.h"

// Batched publication for one push wave. One descriptor per local task; a
// single launch replaces the per-task cuStreamWriteValue32 metadata memops
// and the per-task record kernel, which were the dominant per-task host cost
// of the retire path (issue #208).
extern "C" __global__ void dt_push_completion_batch_k(
    const DtPushCompletionDesc* descs,
    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const DtPushCompletionDesc d = descs[i];
  // Reservation metadata words: nonce, valid length, root token first, then a
  // system-scope release publishes the generation word last, matching the
  // per-task cuStreamWriteValue32 sequence this replaces (the Target CPU
  // treats that generation as its acquire edge).
  {
    unsigned* meta = reinterpret_cast<unsigned*>(d.meta);
    meta[DT_PUSH_META_NONCE_LOW_WORD] = static_cast<unsigned>(d.seq);
    meta[DT_PUSH_META_NONCE_HIGH_WORD] = static_cast<unsigned>(d.seq >> 32);
    meta[DT_PUSH_META_VALID_LENGTH_WORD] = d.valid_length;
    meta[DT_PUSH_META_ROOT_TOKEN_WORD] = d.root_token;
    __threadfence_system();
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(meta),
                 "r"(d.generation)
                 : "memory");
  }
  if (d.kind == DT_PUSH_KIND_COMPLETION) {
    std::uint64_t* record = reinterpret_cast<std::uint64_t*>(d.record);
    std::uint64_t produced_ns;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(produced_ns));
    record[DT_PUSH_RECORD_META_WORD] = std::uint64_t(d.generation) |
                                       (std::uint64_t(d.valid_length) << 32);
    record[DT_PUSH_RECORD_HASH_WORD] = d.record_hash;
    record[DT_PUSH_RECORD_TOKEN_WORD] = std::uint64_t(d.root_token) |
                                        (std::uint64_t(d.producer_tag) << 32);
    record[DT_PUSH_RECORD_PRODUCED_WORD] = produced_ns;
    record[DT_PUSH_RECORD_OFFSET_WORD] = d.globaltimer_offset_ns;
    for (std::uint32_t w = DT_PUSH_RECORD_OFFSET_WORD + 1;
         w < DT_PUSH_RECORD_WORDS;
         ++w) {
      record[w] = 0;
    }
    __threadfence_system();
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(record),
                 "l"(d.seq)
                 : "memory");
  } else if (d.kind == DT_PUSH_KIND_READY) {
    unsigned* record = reinterpret_cast<unsigned*>(d.record);
    record[DT_PUSH_META_NONCE_LOW_WORD] = static_cast<unsigned>(d.seq);
    record[DT_PUSH_META_NONCE_HIGH_WORD] = static_cast<unsigned>(d.seq >> 32);
    record[DT_PUSH_META_VALID_LENGTH_WORD] = d.valid_length;
    record[DT_PUSH_META_ROOT_TOKEN_WORD] = d.root_token;
    record[DT_PUSH_META_PRODUCER_TAG_WORD] = d.producer_tag;
    __threadfence_system();
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(record),
                 "r"(d.generation)
                 : "memory");
  }
}

extern "C" int dt_push_completion_batch_launch(
    int device,
    const void* descs,
    int n,
    std::uint64_t stream) {
  if (n <= 0) {
    return 0;
  }
  cudaError_t err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  constexpr int kThreads = 128;
  int blocks = (n + kThreads - 1) / kThreads;
  dt_push_completion_batch_k<<<blocks, kThreads, 0,
                               reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const DtPushCompletionDesc*>(descs), n);
  return static_cast<int>(cudaGetLastError());
}

// Calibration support: sample %globaltimer into one mapped u64 so the host
// can estimate the CLOCK_REALTIME offset of the GPU timebase.
extern "C" __global__ void dt_globaltimer_k(std::uint64_t* out) {
  std::uint64_t now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  out[0] = now;
}

extern "C" int dt_push_completion_sample_globaltimer(
    int device,
    void* out_device_ptr,
    std::uint64_t stream) {
  cudaError_t err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  dt_globaltimer_k<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<std::uint64_t*>(out_device_ptr));
  return static_cast<int>(cudaGetLastError());
}
