"""CPU contract of the native slot-ready sender's record image (issue #313).

Behind DT_TARGET_READY_SENDER=native the cross-host slot-ready records are
encoded and posted by csrc/dt_target_sender.cpp instead of the Python
publisher thread; the Draft's ring worker and puller validate seq, pool tag
and session tag byte-for-byte against dt_rdma.encode_ready_record, so the
two encoders must stay byte-identical.  A change to either side's record
layout breaks here.  No CUDA, no RDMA: the hook calls the encoder alone.
"""

import random

import numpy as np

from sglang.srt.speculative.dt_rdma import (
    READY_RECORD_BYTES,
    _sender_lib,
    encode_ready_record,
)


def test_native_ready_record_matches_the_python_encoder():
    lib = _sender_lib()
    rng = random.Random(313)
    for case in range(64):
        seq = rng.randrange(1, 1 << 63)
        pool_tag = rng.getrandbits(64)
        session_tag = rng.getrandbits(64)
        slot_id = rng.randrange(0, 1 << 31)
        generation = rng.randrange(1, 0x80000000)
        expected = encode_ready_record(
            seq=seq,
            pool_tag=pool_tag,
            session_tag=session_tag,
            slot_id=slot_id,
            slot_generation=generation,
        )
        out = np.zeros(64, dtype=np.uint8)
        size = lib.dt_target_sender_encode_record(
            out.ctypes.data, seq, pool_tag, session_tag, slot_id, generation
        )
        assert size == READY_RECORD_BYTES, case
        assert out[:size].tobytes() == expected, case
