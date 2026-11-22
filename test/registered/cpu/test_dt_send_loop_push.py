"""CPU contract of the D9 send loop's native publication words (issue #278).

Under DT_DRAFT_SEND_LOOP the send loop pushes a late same-host row itself at
the batch's GPU done: it builds the row's metadata and completion-record
words natively (csrc/dt_publish_worker.cpp ``publication_words``), a port of
``DraftCandidatePushTransport._publication_words``, which still publishes
every pre-queued row.  The Target's scans acquire on those words in that
order (the metadata generation last, the record seq word last of all), and
the record carries the nonce, generation, transfer hash, root token and
producer tag it authenticates.  The two generators must stay byte-identical,
so a change to either side's words, values or order breaks here (design §11
D9).  No CUDA: the hook calls the generator alone.
"""

import ctypes
import random
from types import SimpleNamespace

import numpy as np

from sglang.srt.speculative.dt_candidate_p2p import (
    _RECORD_KIND_COMPLETION,
    DraftCandidatePushTransport,
    _PublishWorker,
    producer_tag,
    transfer_id_hash,
)


def test_native_publication_words_match_the_python_generator():
    lib = _PublishWorker.library()
    u64 = ctypes.c_uint64
    lib.dt_publish_words.argtypes = [ctypes.c_void_p, *(u64,) * 8]
    lib.dt_publish_words.restype = u64
    rng = random.Random(278)
    for case in range(64):
        tag = producer_tag(f"draft-session-{case}")
        transport = SimpleNamespace(_producer_tag=tag)
        transfer_id = f"target-{case}:req-{rng.getrandbits(40):x}:{case + 1}:{case % 3}"
        nonce = rng.randrange(1, 1 << 63)
        generation = rng.randrange(1, 0x80000000)
        row_width = rng.choice((8, 16, 17, 64))
        # A root token above 32 bits is masked to its low word by both.
        root_token = rng.getrandbits(40 if case % 4 == 0 else 20)
        metadata, record = rng.getrandbits(48) & ~3, rng.getrandbits(48) & ~63
        task = SimpleNamespace(transfer_id=transfer_id, next_token_id=root_token)
        destination = SimpleNamespace(
            authorization_nonce=nonce, generation=generation, row_width=row_width
        )
        expected = DraftCandidatePushTransport._publication_words(
            transport,
            [(task, destination, metadata, record, _RECORD_KIND_COMPLETION)],
        )
        out = np.zeros(64, dtype=np.uint64)
        pairs = lib.dt_publish_words(
            out.ctypes.data, metadata, record, nonce, generation, row_width,
            root_token, transfer_id_hash(transfer_id), tag,
        )
        assert out[: 2 * pairs].tolist() == expected, case
