// DT V3 cross-host RDMA verbs shim (issue #165).
//
// A thin C ABI over libibverbs for the CPU-proxy RDMA backend in
// dt_rdma.py: one device context + protection domain, completion queues,
// reliable-connected queue pairs addressed by LID (the DT hosts share one
// InfiniBand subnet), memory registration (host or GPUDirect GPU memory), and
// the three posts the backend uses: zero-SGE receives, inline
// RDMA_WRITE_WITH_IMM records, and RDMA READ.  Every function is called from
// Python through ctypes, which releases the GIL for the call, so the bounded
// spin-then-sleep poll below never stalls the scheduler thread.  No CUDA: the
// GPU visibility fence after a READ lives in Python (cuFlushGPUDirectRDMAWrites).

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <new>

namespace {

struct Device {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  uint8_t port = 1;
  uint16_t lid = 0;
};

// The poll loop spins this long before it starts sleeping between polls.
constexpr uint64_t kSpinNs = 50000;  // 50 us
constexpr long kSleepNs = 20000;     // 20 us

uint64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void set_error(char* err, int errlen, const char* what) {
  if (err != nullptr && errlen > 0) {
    snprintf(err, (size_t)errlen, "%s", what);
  }
}

}  // namespace

extern "C" {

void* dt_rdma_open(const char* name, int port, char* err, int errlen) {
  int count = 0;
  ibv_device** list = ibv_get_device_list(&count);
  if (list == nullptr) {
    set_error(err, errlen, "ibv_get_device_list failed");
    return nullptr;
  }
  ibv_device* chosen = nullptr;
  for (int i = 0; i < count; ++i) {
    if (strcmp(ibv_get_device_name(list[i]), name) == 0) chosen = list[i];
  }
  if (chosen == nullptr) {
    ibv_free_device_list(list);
    set_error(err, errlen, "RDMA device not found");
    return nullptr;
  }
  Device* dev = new (std::nothrow) Device();
  if (dev == nullptr) {
    ibv_free_device_list(list);
    set_error(err, errlen, "out of memory");
    return nullptr;
  }
  dev->ctx = ibv_open_device(chosen);
  ibv_free_device_list(list);
  if (dev->ctx == nullptr) {
    delete dev;
    set_error(err, errlen, "ibv_open_device failed");
    return nullptr;
  }
  ibv_port_attr port_attr;
  if (ibv_query_port(dev->ctx, (uint8_t)port, &port_attr) != 0 ||
      port_attr.state != IBV_PORT_ACTIVE ||
      port_attr.link_layer != IBV_LINK_LAYER_INFINIBAND) {
    ibv_close_device(dev->ctx);
    delete dev;
    set_error(err, errlen, "RDMA port is not an active InfiniBand port");
    return nullptr;
  }
  dev->port = (uint8_t)port;
  dev->lid = port_attr.lid;
  dev->pd = ibv_alloc_pd(dev->ctx);
  if (dev->pd == nullptr) {
    ibv_close_device(dev->ctx);
    delete dev;
    set_error(err, errlen, "ibv_alloc_pd failed");
    return nullptr;
  }
  return dev;
}

int dt_rdma_lid(void* handle) { return ((Device*)handle)->lid; }

void dt_rdma_close(void* handle) {
  Device* dev = (Device*)handle;
  if (dev->pd != nullptr) ibv_dealloc_pd(dev->pd);
  if (dev->ctx != nullptr) ibv_close_device(dev->ctx);
  delete dev;
}

void* dt_rdma_reg_mr(void* handle, uint64_t addr, uint64_t length, int access,
                     uint32_t* lkey, uint32_t* rkey) {
  ibv_mr* mr = ibv_reg_mr(((Device*)handle)->pd, (void*)addr, length, access);
  if (mr == nullptr) return nullptr;
  *lkey = mr->lkey;
  *rkey = mr->rkey;
  return mr;
}

int dt_rdma_dereg_mr(void* mr) { return ibv_dereg_mr((ibv_mr*)mr); }

void* dt_rdma_create_cq(void* handle, int depth) {
  return ibv_create_cq(((Device*)handle)->ctx, depth, nullptr, nullptr, 0);
}

int dt_rdma_destroy_cq(void* cq) { return ibv_destroy_cq((ibv_cq*)cq); }

// Creates an RC QP and moves it to INIT, so receives can be posted before
// the peer's QP number is known.
void* dt_rdma_create_qp(void* handle, void* send_cq, void* recv_cq,
                        int max_send, int max_recv, int max_inline,
                        uint32_t* qpn, uint32_t* max_inline_out) {
  Device* dev = (Device*)handle;
  ibv_qp_init_attr init;
  memset(&init, 0, sizeof(init));
  init.send_cq = (ibv_cq*)send_cq;
  init.recv_cq = (ibv_cq*)recv_cq;
  init.qp_type = IBV_QPT_RC;
  init.cap.max_send_wr = (uint32_t)max_send;
  init.cap.max_recv_wr = (uint32_t)max_recv;
  init.cap.max_send_sge = 1;
  init.cap.max_recv_sge = 1;
  init.cap.max_inline_data = (uint32_t)max_inline;
  ibv_qp* qp = ibv_create_qp(dev->pd, &init);
  if (qp == nullptr) return nullptr;
  ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = dev->port;
  attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  if (ibv_modify_qp(qp, &attr,
                    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                        IBV_QP_ACCESS_FLAGS) != 0) {
    ibv_destroy_qp(qp);
    return nullptr;
  }
  *qpn = qp->qp_num;
  *max_inline_out = init.cap.max_inline_data;
  return qp;
}

// INIT -> RTR -> RTS against the peer's (LID, QP number, PSN).
int dt_rdma_connect_qp(void* qp_handle, int port, uint32_t remote_qpn,
                       uint32_t remote_lid, uint32_t local_psn,
                       uint32_t remote_psn) {
  ibv_qp* qp = (ibv_qp*)qp_handle;
  ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_4096;
  attr.dest_qp_num = remote_qpn;
  attr.rq_psn = remote_psn & 0xFFFFFF;
  attr.max_dest_rd_atomic = 16;
  attr.min_rnr_timer = 1;  // 0.01 ms
  attr.ah_attr.is_global = 0;
  attr.ah_attr.dlid = (uint16_t)remote_lid;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = (uint8_t)port;
  if (ibv_modify_qp(qp, &attr,
                    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
    return -1;
  }
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;   // 4.096 us * 2^14 = 67 ms local ACK timeout
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;  // infinite: a receiver out of receives is backpressure
  attr.sq_psn = local_psn & 0xFFFFFF;
  attr.max_rd_atomic = 16;
  if (ibv_modify_qp(qp, &attr,
                    IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                        IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
    return -2;
  }
  return 0;
}

int dt_rdma_destroy_qp(void* qp_handle) {
  return ibv_destroy_qp((ibv_qp*)qp_handle);
}

// Zero-SGE receives: RDMA_WRITE_WITH_IMM lands its payload at the address
// named by the writer and consumes one receive only for the immediate.
int dt_rdma_post_recv(void* qp_handle, uint64_t wr_id, int count) {
  for (int i = 0; i < count; ++i) {
    ibv_recv_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = nullptr;
    wr.num_sge = 0;
    ibv_recv_wr* bad = nullptr;
    int rc = ibv_post_recv((ibv_qp*)qp_handle, &wr, &bad);
    if (rc != 0) return rc;
  }
  return 0;
}

int dt_rdma_post_write_imm_inline(void* qp_handle, uint64_t wr_id,
                                  const void* data, uint32_t length,
                                  uint64_t remote_addr, uint32_t rkey,
                                  uint32_t imm) {
  ibv_sge sge;
  sge.addr = (uint64_t)data;
  sge.length = length;
  sge.lkey = 0;  // inline data needs no memory registration
  ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  wr.imm_data = htonl(imm);
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = rkey;
  ibv_send_wr* bad = nullptr;
  return ibv_post_send((ibv_qp*)qp_handle, &wr, &bad);
}

int dt_rdma_post_read(void* qp_handle, uint64_t wr_id, uint64_t local_addr,
                      uint32_t length, uint32_t lkey, uint64_t remote_addr,
                      uint32_t rkey) {
  ibv_sge sge;
  sge.addr = local_addr;
  sge.length = length;
  sge.lkey = lkey;
  ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = rkey;
  ibv_send_wr* bad = nullptr;
  return ibv_post_send((ibv_qp*)qp_handle, &wr, &bad);
}

// Polls up to `max` completions, waiting at most `timeout_ns` for the
// first one (spin kSpinNs, then sleep kSleepNs between polls).  Returns the
// number of completions written to the output arrays, or -1 on a poll error.
int dt_rdma_poll(void* cq_handle, int max, uint64_t timeout_ns,
                 uint64_t* wr_id, uint32_t* status, uint32_t* opcode,
                 uint32_t* imm, uint32_t* qpn) {
  ibv_cq* cq = (ibv_cq*)cq_handle;
  ibv_wc wc[64];
  if (max > 64) max = 64;
  const uint64_t start = now_ns();
  for (;;) {
    int n = ibv_poll_cq(cq, max, wc);
    if (n < 0) return -1;
    if (n > 0) {
      for (int i = 0; i < n; ++i) {
        wr_id[i] = wc[i].wr_id;
        status[i] = (uint32_t)wc[i].status;
        opcode[i] = (uint32_t)wc[i].opcode;
        imm[i] = (wc[i].wc_flags & IBV_WC_WITH_IMM) ? ntohl(wc[i].imm_data) : 0;
        qpn[i] = wc[i].qp_num;
      }
      return n;
    }
    const uint64_t elapsed = now_ns() - start;
    if (elapsed >= timeout_ns) return 0;
    if (elapsed >= kSpinNs) {
      timespec pause = {0, kSleepNs};
      nanosleep(&pause, nullptr);
    }
  }
}

}  // extern "C"
