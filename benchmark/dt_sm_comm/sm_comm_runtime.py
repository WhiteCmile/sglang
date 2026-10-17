"""Runtime pieces for experiment A: resident pull kernel, queued push kernel,
C++ pull thread and C++ ring observers.

Shared host region (one /dev/shm file registered in both processes):
  desc ring      RING x 32 B  Target CPU -> Draft (seq written last)
  ready ring     RING x u32   Draft pull mechanism -> Draft CPU (INPUT_READY)
  complete ring  RING x u32   Draft push -> Target CPU (READY)
  control        stop flag, clock calibration slot

All CPU timestamps are CLOCK_MONOTONIC ns, shared by both processes.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import time

import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

RING = 4096
DESC_BYTES = 32
OFF_DESC = 0
OFF_READY = RING * DESC_BYTES
OFF_COMPLETE = OFF_READY + RING * 4
OFF_CTRL = OFF_COMPLETE + RING * 4
REGION_BYTES = 256 * 1024
TMASK = (1 << 20) - 1
MAX_PUSH = 64

_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <sched.h>
#include <time.h>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

namespace {

struct Desc {
  uint32_t seq, batch, bytes, pad;
  uint64_t src_off, dst_off;
};

struct SeqList {
  int n;
  uint32_t seq[64];
};

__device__ __forceinline__ uint32_t ld_acq(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_vol(const uint32_t* p) {
  return *(const volatile uint32_t*)p;
}
__device__ __forceinline__ uint64_t ld_vol64(const uint64_t* p) {
  return *(const volatile uint64_t*)p;
}
__device__ __forceinline__ void st_rel(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ int64_t gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return (int64_t)t;
}

// Each CTA claims the next descriptor seq, waits for it, copies it, publishes READY.
__global__ void resident_pull_k(const Desc* ring, uint32_t* ready, const uint32_t* stop,
                                unsigned int* ticket, const uint8_t* src_base,
                                uint8_t* dst_base, uint32_t ring_size, uint32_t start_seq,
                                int64_t* t_pull, uint32_t tmask) {
  __shared__ uint32_t s_seq, s_bytes;
  __shared__ uint64_t s_src, s_dst;
  __shared__ int s_stop;
  while (true) {
    if (threadIdx.x == 0) {
      const uint32_t seq = start_seq + atomicAdd(ticket, 1u);
      const Desc* d = ring + (seq % ring_size);
      int halt = 0;
      while (ld_acq(&d->seq) != seq) {
        if (ld_vol(stop)) { halt = 1; break; }
      }
      s_stop = halt;
      s_seq = seq;
      if (!halt) {
        s_bytes = ld_vol(&d->bytes);
        s_src = ld_vol64(&d->src_off);
        s_dst = ld_vol64(&d->dst_off);
      }
    }
    __syncthreads();
    if (s_stop) return;
    const uint4* src = (const uint4*)(src_base + s_src);
    uint4* dst = (uint4*)(dst_base + s_dst);
    const uint32_t nv = s_bytes / 16;
    for (uint32_t i = threadIdx.x; i < nv; i += blockDim.x) dst[i] = src[i];
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence_system();
      t_pull[s_seq & tmask] = gtime();
      st_rel(ready + (s_seq % ring_size), s_seq);
    }
    __syncthreads();
  }
}

// Queued right after the proposal: scatter candidates into Target memory, then READY.
__global__ void push_scatter_k(const uint4* cand_src, uint8_t* cand_dst, uint32_t* complete,
                               uint32_t ring_size, SeqList list) {
  const int i = blockIdx.x;
  if (i >= list.n) return;
  const uint32_t seq = list.seq[i];
  const uint32_t slot = seq % ring_size;
  if (threadIdx.x < 2) ((uint4*)(cand_dst + (size_t)slot * 32))[threadIdx.x] = cand_src[i * 2 + threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    st_rel(complete + slot, seq);
  }
}

__global__ void stamp_k(int64_t* t, int64_t idx) { t[idx] = gtime(); }

__global__ void calib_k(int64_t* host_slot) {
  *(volatile int64_t*)host_slot = gtime();
  __threadfence_system();
}

cudaStream_t cur() { return at::cuda::getCurrentCUDAStream().stream(); }

void check_launch(const char* what) {
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

int64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

void pin_cpu(int64_t cpu) {
  if (cpu < 0) return;
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET((int)cpu, &s);
  sched_setaffinity(0, sizeof(s), &s);
}

using WriteValueFn = int (*)(void*, unsigned long long, unsigned int, unsigned int);

WriteValueFn write_value_fn() {
  static WriteValueFn fn = nullptr;
  if (!fn) {
    void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libcuda.so.1", RTLD_NOW);
    TORCH_CHECK(h, "libcuda.so.1 not loadable");
    fn = (WriteValueFn)dlsym(h, "cuStreamWriteValue32_v2");
    TORCH_CHECK(fn, "cuStreamWriteValue32_v2 missing");
  }
  return fn;
}

struct Worker {
  std::atomic<bool> stop{false};
  std::thread th;
};
std::vector<std::unique_ptr<Worker>> g_workers;

}  // namespace

void resident_pull(int64_t ring, int64_t ready, int64_t stop, int64_t ticket, int64_t src_base,
                   int64_t dst_base, int64_t start_seq, torch::Tensor t_pull, int64_t blocks,
                   int64_t threads, int64_t ring_size, int64_t tmask) {
  resident_pull_k<<<blocks, threads, 0, cur()>>>(
      (const Desc*)ring, (uint32_t*)ready, (const uint32_t*)stop, (unsigned int*)ticket,
      (const uint8_t*)src_base, (uint8_t*)dst_base, (uint32_t)ring_size, (uint32_t)start_seq,
      t_pull.data_ptr<int64_t>(), (uint32_t)tmask);
  check_launch("resident_pull");
}

void push_scatter(int64_t cand_src, int64_t cand_dst, int64_t complete, std::vector<int64_t> seqs,
                  int64_t ring_size) {
  SeqList list;
  TORCH_CHECK(seqs.size() <= 64, "too many seqs");
  list.n = (int)seqs.size();
  for (size_t i = 0; i < seqs.size(); ++i) list.seq[i] = (uint32_t)seqs[i];
  push_scatter_k<<<list.n, 32, 0, cur()>>>((const uint4*)cand_src, (uint8_t*)cand_dst,
                                          (uint32_t*)complete, (uint32_t)ring_size, list);
  check_launch("push_scatter");
}

void stamp(torch::Tensor t, int64_t idx) {
  stamp_k<<<1, 1, 0, cur()>>>(t.data_ptr<int64_t>(), idx);
  check_launch("stamp");
}

// Returns (cpu_ns_observed - gpu_globaltimer_ns) from the tightest of `reps` samples.
int64_t clock_offset(int64_t host_slot_dev, int64_t host_slot_cpu, int64_t reps) {
  volatile int64_t* slot = (volatile int64_t*)host_slot_cpu;
  int64_t best_rtt = INT64_MAX, best_off = 0;
  for (int r = 0; r < reps; ++r) {
    *slot = 0;
    const int64_t t0 = now_ns();
    calib_k<<<1, 1, 0, cur()>>>((int64_t*)host_slot_dev);
    int64_t g;
    while ((g = *slot) == 0) {
    }
    const int64_t t1 = now_ns();
    if (t1 - t0 < best_rtt) {
      best_rtt = t1 - t0;
      best_off = t1 - g;
    }
  }
  cudaStreamSynchronize(cur());
  return best_off;
}

int64_t start_pull_thread(int64_t ring_cpu, int64_t ready_dev, int64_t src_base, int64_t dst_base,
                          int64_t start_seq, int64_t device, int64_t cpu, int64_t ring_size) {
  auto w = std::make_unique<Worker>();
  Worker* p = w.get();
  WriteValueFn wv = write_value_fn();
  p->th = std::thread([=] {
    pin_cpu(cpu);
    cudaSetDevice((int)device);
    int lo = 0, hi = 0;
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    cudaStream_t st;
    cudaStreamCreateWithPriority(&st, cudaStreamNonBlocking, hi);
    const volatile Desc* ring = (const volatile Desc*)ring_cpu;
    uint32_t seq = (uint32_t)start_seq;
    while (!p->stop.load(std::memory_order_relaxed)) {
      const volatile Desc* d = ring + (seq % ring_size);
      if (d->seq != seq) continue;
      std::atomic_thread_fence(std::memory_order_acquire);
      cudaMemcpyAsync((void*)(dst_base + d->dst_off), (const void*)(src_base + d->src_off), d->bytes,
                      cudaMemcpyDefault, st);
      wv(st, (unsigned long long)(ready_dev + 4ull * (seq % ring_size)), seq, 0);
      ++seq;
    }
    cudaStreamSynchronize(st);
    cudaStreamDestroy(st);
  });
  g_workers.push_back(std::move(w));
  return (int64_t)g_workers.size() - 1;
}

// Scans a u32 ring and stamps CLOCK_MONOTONIC into times[value & tmask] on every change.
int64_t start_observer(int64_t ring_cpu, int64_t times_ptr, int64_t cpu, int64_t ring_size,
                       int64_t tmask) {
  auto w = std::make_unique<Worker>();
  Worker* p = w.get();
  p->th = std::thread([=] {
    pin_cpu(cpu);
    const volatile uint32_t* ring = (const volatile uint32_t*)ring_cpu;
    int64_t* times = (int64_t*)times_ptr;
    std::vector<uint32_t> last(ring_size);
    for (int64_t s = 0; s < ring_size; ++s) last[s] = ring[s];
    while (!p->stop.load(std::memory_order_relaxed)) {
      for (int64_t s = 0; s < ring_size; ++s) {
        const uint32_t v = ring[s];
        if (v != last[s]) {
          last[s] = v;
          if (v) times[v & tmask] = now_ns();
        }
      }
    }
  });
  g_workers.push_back(std::move(w));
  return (int64_t)g_workers.size() - 1;
}

void stop_worker(int64_t handle) {
  auto& w = g_workers.at(handle);
  if (w->th.joinable()) {
    w->stop.store(true);
    w->th.join();
  }
}

int64_t monotonic_ns() { return now_ns(); }
"""

_DECL = r"""
void resident_pull(int64_t ring, int64_t ready, int64_t stop, int64_t ticket, int64_t src_base,
                   int64_t dst_base, int64_t start_seq, torch::Tensor t_pull, int64_t blocks,
                   int64_t threads, int64_t ring_size, int64_t tmask);
void push_scatter(int64_t cand_src, int64_t cand_dst, int64_t complete, std::vector<int64_t> seqs,
                  int64_t ring_size);
void stamp(torch::Tensor t, int64_t idx);
int64_t clock_offset(int64_t host_slot_dev, int64_t host_slot_cpu, int64_t reps);
int64_t start_pull_thread(int64_t ring_cpu, int64_t ready_dev, int64_t src_base, int64_t dst_base,
                          int64_t start_seq, int64_t device, int64_t cpu, int64_t ring_size);
int64_t start_observer(int64_t ring_cpu, int64_t times_ptr, int64_t cpu, int64_t ring_size,
                       int64_t tmask);
void stop_worker(int64_t handle);
int64_t monotonic_ns();
"""

_FUNCS = ["resident_pull", "push_scatter", "stamp", "clock_offset", "start_pull_thread",
          "start_observer", "stop_worker", "monotonic_ns"]


def load_runtime(build_dir: str):
    os.makedirs(build_dir, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    return load_inline(
        name="dt_sm_comm_runtime",
        cpp_sources=_DECL,
        cuda_sources=_SRC,
        functions=_FUNCS,
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"],
        extra_ldflags=["-ldl"],
        verbose=False,
    )


class Region:
    """The shared host region, mapped and registered in this process."""

    def __init__(self, path: str, drv, create: bool) -> None:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        if create:
            os.ftruncate(fd, REGION_BYTES)
        self._mm = mmap.mmap(fd, REGION_BYTES)
        os.close(fd)
        self.buf = np.frombuffer(self._mm, dtype=np.uint8)
        self.cpu_base = self.buf.ctypes.data
        rc = drv.lib.cuMemHostRegister_v2(ctypes.c_void_p(self.cpu_base), REGION_BYTES, 0x3)
        if rc != 0:
            raise RuntimeError(f"cuMemHostRegister rc={rc}")
        dptr = ctypes.c_uint64()
        rc = drv.lib.cuMemHostGetDevicePointer_v2(ctypes.byref(dptr), ctypes.c_void_p(self.cpu_base), 0)
        if rc != 0:
            raise RuntimeError(f"cuMemHostGetDevicePointer rc={rc}")
        self.dev_base = dptr.value
        self.desc = self.buf[OFF_DESC:OFF_READY].view(np.uint32).reshape(RING, 8)
        self.desc64 = self.buf[OFF_DESC:OFF_READY].view(np.uint64).reshape(RING, 4)
        self.ready = self.buf[OFF_READY:OFF_COMPLETE].view(np.uint32)
        self.complete = self.buf[OFF_COMPLETE:OFF_CTRL].view(np.uint32)
        self.ctrl = self.buf[OFF_CTRL:OFF_CTRL + 64].view(np.uint32)
        self.ctrl64 = self.buf[OFF_CTRL:OFF_CTRL + 64].view(np.int64)

    def dev(self, off: int) -> int:
        return self.dev_base + off

    def cpu(self, off: int) -> int:
        return self.cpu_base + off

    # Control words: ctrl[0] stop flag; ctrl64[4] Draft clock calibration slot;
    # ctrl64[5] Target clock calibration slot.
    @property
    def stop_dev(self) -> int:
        return self.dev(OFF_CTRL)

    def calib_slot(self, which: int) -> tuple[int, int]:
        off = OFF_CTRL + 8 * (4 + which)
        return self.dev(off), self.cpu(off)

    def publish(self, seq: int, batch: int, nbytes: int, src_off: int, dst_off: int) -> None:
        slot = seq % RING
        row = self.desc[slot]
        row[1] = batch
        row[2] = nbytes
        self.desc64[slot, 2] = src_off
        self.desc64[slot, 3] = dst_off
        # x86 stores are not reordered with other stores: seq becomes visible last.
        row[0] = seq

    def reset(self) -> None:
        self.buf[:OFF_CTRL + 64] = 0


def busy_wait_ms(ms: float) -> None:
    end = time.perf_counter() + ms / 1e3
    while time.perf_counter() < end:
        pass


def torch_int64(n: int, device) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.int64, device=device)
