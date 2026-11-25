import multiprocessing as mp, os, sys, time, pickle
sys.path.insert(0, sys.argv[1])
import dt_tp_control_channel as c

def peer(path, n, q):
    ch = c.TPControlChannel(path, 1, 2, create=False)
    q.put("mapped")
    bad = 0; fb = 0
    for i in range(n):
        ok, p = ch.recv()
        if not ok: fb += 1; continue
        if p[0] != i or len(p[1]) != (i % 7) * 100: bad += 1
    q.put((bad, fb))

if __name__ == "__main__":
    path = "/dev/shm/dt-tp-control-test-%d" % os.getpid()
    ch = c.TPControlChannel(path, 0, 2, create=True)
    n = 20000; q = mp.Queue()
    pr = mp.Process(target=peer, args=(path, n, q)); pr.start()
    assert q.get(timeout=30) == "mapped"; os.unlink(path)
    t0 = time.perf_counter(); over = 0
    for i in range(n):
        big = (i % 5000 == 4999)
        payload = (i, b"x" * ((i % 7) * 100) if not big else b"y" * (2 << 20))
        if not ch.send(payload): over += 1
    bad, fb = q.get(); pr.join()
    dt = (time.perf_counter() - t0) / n * 1e6
    print("packets %d bad %d oversize sent %d fallback seen %d  %.1f us/packet (pipelined)" % (n, bad, over, fb, dt))
    # ping-pong latency: producer waits for ack each time (lock-step like the scheduler)
    path2 = path + "b"; ch2 = c.TPControlChannel(path2, 0, 2, create=True)
    pr = mp.Process(target=peer, args=(path2, 5000, q)); pr.start(); assert q.get(timeout=30) == "mapped"; os.unlink(path2)
    lat = []
    for i in range(5000):
        t = time.perf_counter_ns(); ch2.send((i, b"x" * ((i % 7) * 100)))
        while c._WORD.unpack_from(ch2._map, ch2._ack_offset(1))[0] < i + 1: pass
        lat.append(time.perf_counter_ns() - t)
    q.get(); pr.join(); lat.sort()
    print("send->peer ack round trip us p50 %.1f p90 %.1f" % (lat[2500] / 1e3, lat[4500] / 1e3))
    # timeout: no peer
    c.WAIT_TIMEOUT_S = 0.5
    path3 = path + "c"; ch3 = c.TPControlChannel(path3, 1, 2, create=True); os.unlink(path3)
    t = time.monotonic()
    try: ch3.recv(); print("timeout FAILED")
    except TimeoutError as e: print("timeout ok after %.2fs" % (time.monotonic() - t))
