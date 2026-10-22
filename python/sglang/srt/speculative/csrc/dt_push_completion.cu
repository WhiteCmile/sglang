// One-block completion record publication for Draft->Target candidate pushes.
// Launched on the push stream after the candidate copy: the payload words are
// written first, then a system-scope release store publishes the seq word
// last, so a Target CPU that observes the seq word also observes every
// payload word.  The seq word carries the reservation's authorization nonce
// (unique per reservation, known to both sides), so a stale record from a
// prior reservation or producer can never pass the Target's scan.
//
// Record layout per slot (64 bytes, naturally aligned):
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

extern "C" __global__ void dt_push_completion_k(
    std::uint64_t* record,
    std::uint64_t seq,
    std::uint32_t generation,
    std::uint32_t valid_length,
    std::uint64_t transfer_id_hash,
    std::uint32_t root_token,
    std::uint32_t producer_tag,
    std::uint64_t globaltimer_offset_ns) {
  std::uint64_t produced_ns;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(produced_ns));
  record[1] = std::uint64_t(generation) | (std::uint64_t(valid_length) << 32);
  record[2] = transfer_id_hash;
  record[3] = std::uint64_t(root_token) | (std::uint64_t(producer_tag) << 32);
  record[4] = produced_ns;
  record[5] = globaltimer_offset_ns;
  record[6] = 0;
  record[7] = 0;
  __threadfence_system();
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(record), "l"(seq)
               : "memory");
}

extern "C" int dt_push_completion_launch(
    int device,
    void* record_device_ptr,
    std::uint64_t seq,
    std::uint32_t generation,
    std::uint32_t valid_length,
    std::uint64_t transfer_id_hash,
    std::uint32_t root_token,
    std::uint32_t producer_tag,
    std::uint64_t globaltimer_offset_ns,
    std::uint64_t stream) {
  cudaError_t err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  dt_push_completion_k<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<std::uint64_t*>(record_device_ptr),
      seq,
      generation,
      valid_length,
      transfer_id_hash,
      root_token,
      producer_tag,
      globaltimer_offset_ns);
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
