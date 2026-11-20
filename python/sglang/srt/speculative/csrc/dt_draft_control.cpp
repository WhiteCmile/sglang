// Draft control thread (Draft redesign D5b, issue #288; design §8.1, §9 R4).
//
// A native thread owns the Draft's Coordinator DEALER socket once the
// scheduler has registered it.  It never touches a Python object and never
// takes the GIL: it moves raw frames only.
// - Inbound: every frame read from the socket is copied into the inbound
//   SPSC ring (this thread produces, the scheduler consumes), then the
//   scheduler's wake word gets ``reason`` once per received batch.  When the
//   ring is full the socket is left unread (its receive high-water mark then
//   pushes back on the Coordinator) and the thread re-checks every 1 ms.
// - Outbound: the scheduler copies each encoded frame into the outbound SPSC
//   ring and signals an eventfd; frames leave in ring order.  A frame refused
//   at the socket's send high-water mark stays at the head until the socket
//   polls writable.
// - Heartbeat: a frame pre-encoded by the scheduler, sent on this thread's
//   timer.  A refused heartbeat is counted, not retried (the next tick sends
//   again; the Coordinator's lease expiry is the failure detector).
// - A failing libzmq call ends the thread with its errno, after a wake, so
//   the scheduler raises it at its next call.
// libzmq is resolved through pyzmq's already-loaded backend extension, so the
// symbols are those of the copy that created the socket even when another
// libzmq is loaded too; this file needs no zmq headers: the few ABI items it
// uses are declared below as zmq.h 4.x defines them.

#include <dlfcn.h>
#include <errno.h>
#include <sys/eventfd.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

struct ZmqPollItem {  // zmq_pollitem_t (non-Windows)
  void* socket;
  int fd;
  short events;
  short revents;
};
struct ZmqMsg {  // zmq_msg_t
  alignas(sizeof(void*)) unsigned char bytes[64];
};
constexpr short kPollIn = 1;   // ZMQ_POLLIN
constexpr short kPollOut = 2;  // ZMQ_POLLOUT
constexpr int kDontWait = 1;   // ZMQ_DONTWAIT

struct Zmq {
  int (*poll)(ZmqPollItem*, int, long);
  int (*send)(void*, const void*, size_t, int);
  int (*msg_init)(ZmqMsg*);
  int (*msg_recv)(ZmqMsg*, void*, int);
  void* (*msg_data)(ZmqMsg*);
  size_t (*msg_size)(const ZmqMsg*);
  int (*msg_close)(ZmqMsg*);
  int (*errno_)();
};

// Single-producer single-consumer ring of heap frames.  ``head`` is written
// by the producer only and ``tail`` by the consumer only (release stores,
// acquire loads); a slot is reused only after the consumer moved past it.
struct Ring {
  explicit Ring(uint64_t capacity) : slots(capacity, nullptr) {}
  ~Ring() {
    for (std::string* frame : slots) delete frame;
  }
  uint64_t size() const {
    return head.load(std::memory_order_acquire) -
           tail.load(std::memory_order_acquire);
  }
  std::vector<std::string*> slots;
  alignas(64) std::atomic<uint64_t> head{0};
  alignas(64) std::atomic<uint64_t> tail{0};
};

enum Stop : uint32_t { kRun = 0, kStop = 1 };

struct Control {
  Control(uint64_t out_capacity, uint64_t in_capacity)
      : out(out_capacity), in(in_capacity) {}
  Zmq z{};
  void* socket = nullptr;
  int signal_fd = -1;
  void (*wake_set)(void*, uint32_t) = nullptr;
  void* wake = nullptr;
  uint32_t reason = 0;
  Ring out;  // scheduler -> socket
  Ring in;   // socket -> scheduler
  // Written by the scheduler only while the thread is stopped.
  std::string heartbeat;
  int64_t heartbeat_interval_ns = 0;
  std::thread thread;
  std::atomic<uint32_t> stop{kRun};
  std::atomic<int> error{0};  // errno of the failed libzmq call
  std::atomic<uint64_t> heartbeat_refused{0};
};

int64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return int64_t(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

void fail(Control* c, int err) {
  c->error.store(err ? err : EIO, std::memory_order_release);
  c->wake_set(c->wake, c->reason);
}

// Send the outbound ring's frames until it is empty or the socket refuses
// one at its high-water mark.  False on a libzmq failure.
bool drain_out(Control* c) {
  Ring& r = c->out;
  uint64_t tail = r.tail.load(std::memory_order_relaxed);
  while (tail != r.head.load(std::memory_order_acquire)) {
    std::string*& slot = r.slots[tail % r.slots.size()];
    if (c->z.send(c->socket, slot->data(), slot->size(), kDontWait) < 0) {
      int err = c->z.errno_();
      if (err == EINTR) continue;
      if (err == EAGAIN) return true;
      fail(c, err);
      return false;
    }
    delete slot;
    slot = nullptr;
    r.tail.store(++tail, std::memory_order_release);
  }
  return true;
}

// Read frames while the inbound ring has room.  False on a libzmq failure.
bool fill_in(Control* c, bool* received) {
  Ring& r = c->in;
  uint64_t head = r.head.load(std::memory_order_relaxed);
  while (head - r.tail.load(std::memory_order_acquire) < r.slots.size()) {
    ZmqMsg msg;
    c->z.msg_init(&msg);
    if (c->z.msg_recv(&msg, c->socket, kDontWait) < 0) {
      int err = c->z.errno_();
      c->z.msg_close(&msg);
      if (err == EINTR) continue;
      if (err == EAGAIN) return true;
      fail(c, err);
      return false;
    }
    r.slots[head % r.slots.size()] = new std::string(
        static_cast<const char*>(c->z.msg_data(&msg)), c->z.msg_size(&msg));
    c->z.msg_close(&msg);
    r.head.store(++head, std::memory_order_release);
    *received = true;
  }
  return true;
}

void run(Control* c) {
  int64_t next_heartbeat_ns = now_ns() + c->heartbeat_interval_ns;
  while (true) {
    bool in_full = c->in.size() >= c->in.slots.size();
    ZmqPollItem items[2] = {
        {c->socket, 0,
         short((in_full ? 0 : kPollIn) | (c->out.size() ? kPollOut : 0)), 0},
        {nullptr, c->signal_fd, kPollIn, 0},
    };
    long timeout_ms = -1;
    if (c->heartbeat_interval_ns > 0) {
      // Round up: a sub-millisecond remainder must not spin.
      int64_t left = next_heartbeat_ns - now_ns();
      timeout_ms = left > 0 ? long((left + 999999) / 1000000) : 0;
    }
    if (in_full && (timeout_ms < 0 || timeout_ms > 1)) timeout_ms = 1;
    if (c->z.poll(items, 2, timeout_ms) < 0) {
      int err = c->z.errno_();
      if (err == EINTR) continue;
      fail(c, err);
      return;
    }
    // Read before the drain: every frame queued before the stop request is
    // in the ring by then and leaves below.
    uint32_t stop = c->stop.load(std::memory_order_acquire);
    if (items[1].revents) {
      uint64_t count;
      (void)!read(c->signal_fd, &count, sizeof(count));
    }
    if (!drain_out(c)) return;
    if (stop != kRun) return;
    bool received = false;
    if (!fill_in(c, &received)) return;
    if (received) c->wake_set(c->wake, c->reason);
    if (c->heartbeat_interval_ns > 0 && now_ns() >= next_heartbeat_ns) {
      if (c->z.send(c->socket, c->heartbeat.data(), c->heartbeat.size(),
                    kDontWait) < 0) {
        int err = c->z.errno_();
        if (err != EAGAIN && err != EINTR) {
          fail(c, err);
          return;
        }
        c->heartbeat_refused.fetch_add(1, std::memory_order_relaxed);
      }
      next_heartbeat_ns = now_ns() + c->heartbeat_interval_ns;
    }
  }
}

template <typename F>
bool resolve(void* lib, const char* name, F* out) {
  *out = reinterpret_cast<F>(dlsym(lib, name));
  return *out != nullptr;
}

}  // namespace

extern "C" {

// ``zmq_user`` names an already-loaded library linked against the libzmq that
// created ``socket`` (dlsym searches its dependency tree).  Returns nullptr
// when it cannot be resolved.  The thread starts with ``dt_ctl_start``.
void* dt_ctl_create(const char* zmq_user, void* socket, void* wake_set,
                    void* wake, uint32_t reason, uint64_t out_capacity,
                    uint64_t in_capacity) {
  void* lib = dlopen(zmq_user, RTLD_NOW | RTLD_NOLOAD);
  if (lib == nullptr || out_capacity == 0 || in_capacity == 0) return nullptr;
  auto* c = new Control(out_capacity, in_capacity);
  bool ok = resolve(lib, "zmq_poll", &c->z.poll) &&
            resolve(lib, "zmq_send", &c->z.send) &&
            resolve(lib, "zmq_msg_init", &c->z.msg_init) &&
            resolve(lib, "zmq_msg_recv", &c->z.msg_recv) &&
            resolve(lib, "zmq_msg_data", &c->z.msg_data) &&
            resolve(lib, "zmq_msg_size", &c->z.msg_size) &&
            resolve(lib, "zmq_msg_close", &c->z.msg_close) &&
            resolve(lib, "zmq_errno", &c->z.errno_);
  c->signal_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (!ok || c->signal_fd < 0) {
    if (c->signal_fd >= 0) close(c->signal_fd);
    delete c;
    return nullptr;
  }
  c->socket = socket;
  c->wake_set = reinterpret_cast<void (*)(void*, uint32_t)>(wake_set);
  c->wake = wake;
  c->reason = reason;
  return c;
}

// Start (or restart after dt_ctl_stop) the thread; the socket belongs to it
// until dt_ctl_stop returns.  ``heartbeat_interval_ns`` 0 sends none.
void dt_ctl_start(void* handle, const char* heartbeat, uint64_t heartbeat_len,
                  int64_t heartbeat_interval_ns) {
  auto* c = static_cast<Control*>(handle);
  c->heartbeat.assign(heartbeat, heartbeat_len);
  c->heartbeat_interval_ns = heartbeat_interval_ns;
  c->stop.store(kRun, std::memory_order_relaxed);
  c->thread = std::thread(run, c);
}

// Send the frames queued so far (without waiting at the send high-water
// mark), stop and join the thread; the socket is the caller's again.
// Returns the thread's errno, 0 if it did not fail.
int dt_ctl_stop(void* handle) {
  auto* c = static_cast<Control*>(handle);
  if (c->thread.joinable()) {
    c->stop.store(kStop, std::memory_order_release);
    uint64_t one = 1;
    (void)!write(c->signal_fd, &one, sizeof(one));
    c->thread.join();
  }
  return c->error.load(std::memory_order_acquire);
}

void dt_ctl_destroy(void* handle) {
  auto* c = static_cast<Control*>(handle);
  dt_ctl_stop(c);
  close(c->signal_fd);
  delete c;
}

// Queue one outbound frame.  1 queued, 0 ring full, -errno once the thread
// has failed.
int dt_ctl_send(void* handle, const char* data, uint64_t len) {
  auto* c = static_cast<Control*>(handle);
  if (int err = c->error.load(std::memory_order_acquire)) return -err;
  Ring& r = c->out;
  uint64_t head = r.head.load(std::memory_order_relaxed);
  if (head - r.tail.load(std::memory_order_acquire) >= r.slots.size()) return 0;
  r.slots[head % r.slots.size()] = new std::string(data, len);
  r.head.store(head + 1, std::memory_order_release);
  uint64_t one = 1;
  (void)!write(c->signal_fd, &one, sizeof(one));
  return 1;
}

// The oldest inbound frame: its length, with ``*data`` pointing at it until
// dt_ctl_pop.  -1 when there is none, -2 when there is none and the thread
// has failed (dt_ctl_stop returns its errno).
int64_t dt_ctl_front(void* handle, const char** data) {
  auto* c = static_cast<Control*>(handle);
  Ring& r = c->in;
  uint64_t tail = r.tail.load(std::memory_order_relaxed);
  if (tail == r.head.load(std::memory_order_acquire)) {
    return c->error.load(std::memory_order_acquire) ? -2 : -1;
  }
  const std::string* frame = r.slots[tail % r.slots.size()];
  *data = frame->data();
  return int64_t(frame->size());
}

void dt_ctl_pop(void* handle) {
  auto* c = static_cast<Control*>(handle);
  Ring& r = c->in;
  uint64_t tail = r.tail.load(std::memory_order_relaxed);
  std::string*& slot = r.slots[tail % r.slots.size()];
  delete slot;
  slot = nullptr;
  r.tail.store(tail + 1, std::memory_order_release);
}

uint64_t dt_ctl_heartbeats_refused(void* handle) {
  return static_cast<Control*>(handle)->heartbeat_refused.load(
      std::memory_order_relaxed);
}

}  // extern "C"
