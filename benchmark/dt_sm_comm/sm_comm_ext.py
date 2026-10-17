"""CUDA kernels and driver helpers for the Draft-side SM communication prototype.

Kernels (all payloads are uint4 words stamped with the iteration number):
- push_writer: persistent multi-CTA writer. Per iteration: wait ack == it-1,
  store payload (possibly into peer GPU memory), last CTA publishes flag = it.
- pull_reader: persistent multi-CTA reader. Per iteration: wait desc == it,
  load payload (possibly from peer GPU memory), count stale words, ack = it.
- check / fill / store_flag: one-shot helpers launched on the current stream.

fence=1 uses __threadfence_system + st.release.sys / ld.acquire.sys; fence=0
is the negative control (plain volatile flag access, no fences).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint32_t ld_acq(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_vol(const uint32_t* p) {
  return *(const volatile uint32_t*)p;
}
__device__ __forceinline__ void st_rel(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_vol(uint32_t* p, uint32_t v) {
  *(volatile uint32_t*)p = v;
}
__device__ __forceinline__ uint64_t gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void push_writer_k(uint4* data, int64_t n_vec, uint32_t* flag,
                              const uint32_t* ack, unsigned int* done,
                              uint64_t* t_flag, uint64_t* t_ack, int iters,
                              int fence) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (uint32_t it = 1; it <= (uint32_t)iters; ++it) {
    if (threadIdx.x == 0) {
      while ((fence ? ld_acq(ack) : ld_vol(ack)) != it - 1) {
      }
      if (blockIdx.x == 0 && it > 1) t_ack[it - 2] = gtime();
    }
    __syncthreads();
    const uint4 v = make_uint4(it, it, it, it);
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
         i += stride) {
      data[i] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      if (fence) __threadfence_system();
      unsigned int old = atomicAdd(done, 1u);
      if (old == it * gridDim.x - 1) {
        if (fence) __threadfence_system();
        t_flag[it - 1] = gtime();
        if (fence) st_rel(flag, it); else st_vol(flag, it);
      }
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    while ((fence ? ld_acq(ack) : ld_vol(ack)) != (uint32_t)iters) {
    }
    t_ack[iters - 1] = gtime();
  }
}

__global__ void pull_reader_k(const uint4* src, int64_t n_vec,
                              const uint32_t* desc, uint32_t* ack,
                              unsigned int* done, unsigned long long* err,
                              uint64_t* t_desc, int iters, int fence,
                              int load_mode) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (uint32_t it = 1; it <= (uint32_t)iters; ++it) {
    if (threadIdx.x == 0) {
      while ((fence ? ld_acq(desc) : ld_vol(desc)) != it) {
      }
      if (blockIdx.x == 0) t_desc[it - 1] = gtime();
    }
    __syncthreads();
    unsigned int bad = 0;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
         i += stride) {
      uint4 x;
      if (load_mode == 0) x = src[i];
      else if (load_mode == 1) x = __ldcg(src + i);
      else x = __ldcv(src + i);
      bad += (x.x != it) + (x.y != it) + (x.z != it) + (x.w != it);
    }
    const int nbad = __syncthreads_count(bad != 0);
    if (threadIdx.x == 0) {
      if (nbad) atomicAdd(err, (unsigned long long)nbad);
      if (fence) __threadfence_system();
      unsigned int old = atomicAdd(done, 1u);
      if (old == it * gridDim.x - 1) {
        if (fence) __threadfence_system();
        if (fence) st_rel(ack, it); else st_vol(ack, it);
      }
    }
  }
}

__global__ void check_k(const uint4* data, int64_t n_vec, uint32_t expect,
                        unsigned long long* err) {
  unsigned int bad = 0;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
       i += (int64_t)gridDim.x * blockDim.x) {
    const uint4 x = data[i];
    bad += (x.x != expect) + (x.y != expect) + (x.z != expect) + (x.w != expect);
  }
  const int nbad = __syncthreads_count(bad != 0);
  if (threadIdx.x == 0 && nbad) atomicAdd(err, (unsigned long long)nbad);
}

__global__ void fill_k(uint4* data, int64_t n_vec, uint32_t v) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
       i += (int64_t)gridDim.x * blockDim.x) {
    data[i] = make_uint4(v, v, v, v);
  }
}

__global__ void store_flag_k(uint32_t* p, uint32_t v) {
  __threadfence_system();
  st_rel(p, v);
}

int grid_for(int64_t n_vec, int threads) {
  int64_t g = (n_vec + threads - 1) / threads;
  return (int)(g < 1 ? 1 : (g > 1056 ? 1056 : g));
}

cudaStream_t cur() { return at::cuda::getCurrentCUDAStream().stream(); }

void check_launch(const char* what) {
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

}  // namespace

void push_writer(int64_t data, int64_t n_vec, int64_t flag, int64_t ack,
                 torch::Tensor done, torch::Tensor t_flag, torch::Tensor t_ack,
                 int64_t iters, int64_t fence, int64_t blocks, int64_t threads) {
  push_writer_k<<<blocks, threads, 0, cur()>>>(
      (uint4*)data, n_vec, (uint32_t*)flag, (const uint32_t*)ack,
      (unsigned int*)done.data_ptr<int32_t>(), (uint64_t*)t_flag.data_ptr<int64_t>(),
      (uint64_t*)t_ack.data_ptr<int64_t>(), (int)iters, (int)fence);
  check_launch("push_writer");
}

void pull_reader(int64_t src, int64_t n_vec, int64_t desc, int64_t ack,
                 torch::Tensor done, torch::Tensor err, torch::Tensor t_desc,
                 int64_t iters, int64_t fence, int64_t load_mode, int64_t blocks,
                 int64_t threads) {
  pull_reader_k<<<blocks, threads, 0, cur()>>>(
      (const uint4*)src, n_vec, (const uint32_t*)desc, (uint32_t*)ack,
      (unsigned int*)done.data_ptr<int32_t>(),
      (unsigned long long*)err.data_ptr<int64_t>(),
      (uint64_t*)t_desc.data_ptr<int64_t>(), (int)iters, (int)fence,
      (int)load_mode);
  check_launch("pull_reader");
}

void check(int64_t data, int64_t n_vec, int64_t expect, torch::Tensor err) {
  check_k<<<grid_for(n_vec, 256), 256, 0, cur()>>>(
      (const uint4*)data, n_vec, (uint32_t)expect,
      (unsigned long long*)err.data_ptr<int64_t>());
  check_launch("check");
}

void fill(int64_t data, int64_t n_vec, int64_t v) {
  fill_k<<<grid_for(n_vec, 256), 256, 0, cur()>>>((uint4*)data, n_vec, (uint32_t)v);
  check_launch("fill");
}

void store_flag(int64_t ptr, int64_t v) {
  store_flag_k<<<1, 1, 0, cur()>>>((uint32_t*)ptr, (uint32_t)v);
  check_launch("store_flag");
}

int64_t dev_alloc(int64_t bytes) {
  void* p = nullptr;
  cudaError_t e = cudaMalloc(&p, bytes);
  TORCH_CHECK(e == cudaSuccess, "cudaMalloc: ", cudaGetErrorString(e));
  e = cudaMemset(p, 0, bytes);
  TORCH_CHECK(e == cudaSuccess, "cudaMemset: ", cudaGetErrorString(e));
  return (int64_t)p;
}

torch::Tensor ipc_get(int64_t ptr) {
  cudaIpcMemHandle_t h;
  cudaError_t e = cudaIpcGetMemHandle(&h, (void*)ptr);
  TORCH_CHECK(e == cudaSuccess, "cudaIpcGetMemHandle: ", cudaGetErrorString(e));
  auto out = torch::empty({(int64_t)sizeof(h)}, torch::kUInt8);
  memcpy(out.data_ptr(), &h, sizeof(h));
  return out;
}

int64_t ipc_open(torch::Tensor handle) {
  cudaIpcMemHandle_t h;
  memcpy(&h, handle.data_ptr(), sizeof(h));
  void* p = nullptr;
  cudaError_t e = cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess);
  TORCH_CHECK(e == cudaSuccess, "cudaIpcOpenMemHandle: ", cudaGetErrorString(e));
  return (int64_t)p;
}

void memset_async(int64_t ptr, int64_t bytes) {
  cudaError_t e = cudaMemsetAsync((void*)ptr, 0, bytes, cur());
  TORCH_CHECK(e == cudaSuccess, "cudaMemsetAsync: ", cudaGetErrorString(e));
}

void memcpy_async(int64_t dst, int64_t src, int64_t bytes) {
  cudaError_t e = cudaMemcpyAsync((void*)dst, (const void*)src, bytes,
                                  cudaMemcpyDefault, cur());
  TORCH_CHECK(e == cudaSuccess, "cudaMemcpyAsync: ", cudaGetErrorString(e));
}
"""

_CPP_DECL = r"""
void push_writer(int64_t data, int64_t n_vec, int64_t flag, int64_t ack,
                 torch::Tensor done, torch::Tensor t_flag, torch::Tensor t_ack,
                 int64_t iters, int64_t fence, int64_t blocks, int64_t threads);
void pull_reader(int64_t src, int64_t n_vec, int64_t desc, int64_t ack,
                 torch::Tensor done, torch::Tensor err, torch::Tensor t_desc,
                 int64_t iters, int64_t fence, int64_t load_mode, int64_t blocks,
                 int64_t threads);
void check(int64_t data, int64_t n_vec, int64_t expect, torch::Tensor err);
void fill(int64_t data, int64_t n_vec, int64_t v);
void store_flag(int64_t ptr, int64_t v);
void memcpy_async(int64_t dst, int64_t src, int64_t bytes);
int64_t dev_alloc(int64_t bytes);
torch::Tensor ipc_get(int64_t ptr);
int64_t ipc_open(torch::Tensor handle);
void memset_async(int64_t ptr, int64_t bytes);
"""

_FUNCS = ["push_writer", "pull_reader", "check", "fill", "store_flag", "memcpy_async",
          "dev_alloc", "ipc_get", "ipc_open", "memset_async"]


def load_ext(build_dir: str):
    os.makedirs(build_dir, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    return load_inline(
        name="dt_sm_comm_ext",
        cpp_sources=_CPP_DECL,
        cuda_sources=_CUDA_SRC,
        functions=_FUNCS,
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"],
        verbose=False,
    )


CU_STREAM_WAIT_VALUE_EQ = 0x1
CU_MEMHOSTREGISTER_PORTABLE = 0x01
CU_MEMHOSTREGISTER_DEVICEMAP = 0x02


class Driver:
    """ctypes binding for the stream memory operations."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")
        self._wait = self.lib.cuStreamWaitValue32_v2
        self._write = self.lib.cuStreamWriteValue32_v2
        for fn in (self._wait, self._write):
            fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
            fn.restype = ctypes.c_int
        self.lib.cuMemHostRegister_v2.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self.lib.cuMemHostGetDevicePointer_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint
        ]

    def wait_eq(self, stream: torch.cuda.Stream, addr: int, value: int) -> None:
        rc = self._wait(ctypes.c_void_p(stream.cuda_stream), addr, value, CU_STREAM_WAIT_VALUE_EQ)
        if rc != 0:
            raise RuntimeError(f"cuStreamWaitValue32 rc={rc}")

    def write(self, stream: torch.cuda.Stream, addr: int, value: int) -> int:
        return self._write(ctypes.c_void_p(stream.cuda_stream), addr, value, 0)


class SharedFlags:
    """One /dev/shm page of u32 flags, registered for device access.

    Each flag sits on its own 128-byte line. Both processes map the same file
    and register it in their own CUDA context.
    """

    NAMES = ("flag", "ack", "desc", "ackp")

    def __init__(self, path: str, drv: Driver, create: bool) -> None:
        self.path = path
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        if create:
            os.ftruncate(fd, 4096)
        self._mm = mmap.mmap(fd, 4096)
        os.close(fd)
        self.view = (ctypes.c_uint32 * 1024).from_buffer(self._mm)
        host = ctypes.addressof(self.view)
        rc = drv.lib.cuMemHostRegister_v2(
            ctypes.c_void_p(host), 4096,
            CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP,
        )
        if rc != 0:
            raise RuntimeError(f"cuMemHostRegister rc={rc}")
        dptr = ctypes.c_uint64()
        rc = drv.lib.cuMemHostGetDevicePointer_v2(ctypes.byref(dptr), ctypes.c_void_p(host), 0)
        if rc != 0:
            raise RuntimeError(f"cuMemHostGetDevicePointer rc={rc}")
        self.dev_base = dptr.value

    def index(self, name: str) -> int:
        return self.NAMES.index(name) * 32

    def addr(self, name: str) -> int:
        return self.dev_base + 4 * self.index(name)

    def get(self, name: str) -> int:
        return self.view[self.index(name)]

    def set(self, name: str, value: int) -> None:
        self.view[self.index(name)] = value

    def reset(self) -> None:
        for name in self.NAMES:
            self.set(name, 0)
