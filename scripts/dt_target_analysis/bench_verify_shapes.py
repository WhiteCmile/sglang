"""Per-shape timing of the Qwen3-14B TP2 verify forward pieces on one GPU.

Why: traced verify model time per CUDA-graph bucket is non-monotonic (bs40 = 320 tokens 15.28 ms vs
bs44 13.43 ms; bs24 10.27 vs bs26 9.85).  This times every per-rank GEMM of one TP2 Qwen3-14B layer,
the vocab-parallel lm_head and the FA3 verify attention at each token count M = 8 * rows, so the shape
that is slow can be named.  Usage:
    python bench_verify_shapes.py --out <file.tsv> [--rows 1-64] [--kv-len 160] [--reps 50]
Output TSV columns: rows, tokens, op, us (median of reps, CUDA graph replay of 10 back-to-back calls / 10).
"""
import argparse
import statistics
import sys

import torch
import torch.nn.functional as F

# Qwen3-14B (config.json), tensor parallel 2
HIDDEN, INTER, HEADS, KV_HEADS, HEAD_DIM, LAYERS, VOCAB = 5120, 17408, 40, 8, 128, 40, 151936
TP = 2
BLOCK = 8  # FlashMTP block size (tokens verified per row)


def parse_rows(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def time_fn(fn, reps):
    # Graph-capture 10 calls so launch overhead does not hide kernel time (the real verify is a graph).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / 10)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="1-64")
    ap.add_argument("--kv-len", type=int, default=160, help="KV length per row (prompt ~13 + generated <=128)")
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    dev = torch.device("cuda")
    bf = torch.bfloat16
    torch.manual_seed(0)
    qkv_n = (HEADS + 2 * KV_HEADS) * HEAD_DIM // TP
    w = {
        "qkv": torch.randn(qkv_n, HIDDEN, device=dev, dtype=bf) * 0.02,
        "o": torch.randn(HIDDEN, HEADS * HEAD_DIM // TP, device=dev, dtype=bf) * 0.02,
        "gate_up": torch.randn(2 * INTER // TP, HIDDEN, device=dev, dtype=bf) * 0.02,
        "down": torch.randn(HIDDEN, INTER // TP, device=dev, dtype=bf) * 0.02,
        "lm_head": torch.randn(VOCAB // TP, HIDDEN, device=dev, dtype=bf) * 0.02,
    }
    try:
        from sgl_kernel.flash_attn import flash_attn_varlen_func  # FA3, as used by the fa3 backend
    except Exception as exc:  # keep the GEMM sweep useful even without FA3
        print("FA3 unavailable:", exc, file=sys.stderr)
        flash_attn_varlen_func = None

    rows_list = parse_rows(args.rows)
    with open(args.out, "w") as f:
        f.write("rows\ttokens\top\tus\n")
        for rows in rows_list:
            m = rows * BLOCK
            x_h = torch.randn(m, HIDDEN, device=dev, dtype=bf)
            x_o = torch.randn(m, HEADS * HEAD_DIM // TP, device=dev, dtype=bf)
            x_d = torch.randn(m, INTER // TP, device=dev, dtype=bf)
            res = {}
            res["qkv"] = time_fn(lambda: F.linear(x_h, w["qkv"]), args.reps)
            res["o"] = time_fn(lambda: F.linear(x_o, w["o"]), args.reps)
            res["gate_up"] = time_fn(lambda: F.linear(x_h, w["gate_up"]), args.reps)
            res["down"] = time_fn(lambda: F.linear(x_d, w["down"]), args.reps)
            res["lm_head"] = time_fn(lambda: F.linear(x_h, w["lm_head"]), args.reps)
            if flash_attn_varlen_func is not None:
                h, kvh, kv = HEADS // TP, KV_HEADS // TP, args.kv_len
                q = torch.randn(m, h, HEAD_DIM, device=dev, dtype=bf)
                k = torch.randn(rows * kv, kvh, HEAD_DIM, device=dev, dtype=bf)
                v = torch.randn_like(k)
                cu_q = torch.arange(0, (rows + 1) * BLOCK, BLOCK, device=dev, dtype=torch.int32)
                cu_k = torch.arange(0, (rows + 1) * kv, kv, device=dev, dtype=torch.int32)
                res["attn"] = time_fn(
                    lambda: flash_attn_varlen_func(
                        q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                        max_seqlen_q=BLOCK, max_seqlen_k=kv, causal=True,
                    ),
                    args.reps,
                )
            layer = sum(v for k_, v in res.items() if k_ != "lm_head")
            res["layer_sum"] = layer
            res["model_est"] = layer * LAYERS + res["lm_head"]
            for op, us in res.items():
                f.write(f"{rows}\t{m}\t{op}\t{us:.1f}\n")
            f.flush()
            print(rows, m, " ".join(f"{k_}={v:.0f}" for k_, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
