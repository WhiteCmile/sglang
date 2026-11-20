// Draft scheduler wake word (Draft redesign D5, issue #274; design §4.4).
//
// One word per Draft scheduler.  Any thread may set reason bits; only the
// scheduler waits.  ``epoch`` is the futex word and is bumped by every set, so
// a set between the scheduler's last epoch read and its wait is never lost:
// the wait returns at once when the epoch already moved.  Waiters are rare
// relative to sets, so a set issues the FUTEX_WAKE syscall only while the
// scheduler has announced itself in ``waiting`` (Dekker pair, seq_cst).
// Built and loaded through ctypes (dt_wake.py), which releases the GIL for
// the whole wait.

#include <linux/futex.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <climits>
#include <cstdint>

namespace {

struct Wake {
  std::atomic<uint32_t> epoch{0};    // futex word; +1 per set
  std::atomic<uint32_t> reasons{0};  // OR of the bits set since the last take
  std::atomic<uint32_t> waiting{0};  // 1 while the scheduler may sleep
};

static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t),
              "futex needs a plain 32-bit word");

long futex(std::atomic<uint32_t>* word, int op, uint32_t value,
           const timespec* timeout) {
  return syscall(SYS_futex, reinterpret_cast<uint32_t*>(word), op, value,
                 timeout, nullptr, 0);
}

}  // namespace

extern "C" {

void* dt_wake_create() { return new Wake(); }

void dt_wake_destroy(void* handle) { delete static_cast<Wake*>(handle); }

uint32_t dt_wake_epoch(void* handle) {
  return static_cast<Wake*>(handle)->epoch.load(std::memory_order_seq_cst);
}

void dt_wake_set(void* handle, uint32_t reasons) {
  auto* w = static_cast<Wake*>(handle);
  w->reasons.fetch_or(reasons, std::memory_order_release);
  w->epoch.fetch_add(1, std::memory_order_seq_cst);
  if (w->waiting.load(std::memory_order_seq_cst)) {
    futex(&w->epoch, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr);
  }
}

// Sleep until the epoch differs from ``seen_epoch`` or ``timeout_ns`` passes
// (a signal may end it early), then take and return the pending reason bits
// (0 on a plain timeout).
uint32_t dt_wake_wait(void* handle, uint32_t seen_epoch, int64_t timeout_ns) {
  auto* w = static_cast<Wake*>(handle);
  w->waiting.store(1, std::memory_order_seq_cst);
  if (timeout_ns > 0 &&
      w->epoch.load(std::memory_order_seq_cst) == seen_epoch) {
    timespec timeout{static_cast<time_t>(timeout_ns / 1000000000),
                     static_cast<long>(timeout_ns % 1000000000)};
    futex(&w->epoch, FUTEX_WAIT_PRIVATE, seen_epoch, &timeout);
  }
  w->waiting.store(0, std::memory_order_relaxed);
  return w->reasons.exchange(0, std::memory_order_acq_rel);
}

}  // extern "C"
