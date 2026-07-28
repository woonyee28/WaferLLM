"""verify_rope_fix.py — confirm the decode RoPE fix WITHOUT recompiling (pure numpy + saved dump).

--dump-intermediates showed QKV projection is correct but post-RoPE Q/K are near-orthogonal to HF.
This replays the kernel's per-PE RoPE on the SAVED device pre-rope values
(dump_intermediates/raw_QKV_post_reduce.npy), using the exact host freq layout, under BOTH the
CURRENT kernel formula and the PROPOSED fix, then un-tiles to head-space and compares to the trusted
HF reference (ref_Qr/ref_Kr). No device, no torch, no cerebras — just numpy + the saved .npy files.

Cross-check: the CURRENT-formula cosine should reproduce the device's observed QKV_tile numbers
(~ -0.006 for Q, ~ -0.017 for K). If it does, the emulation is faithful and the FIXED number is
trustworthy — a FIXED cosine ~1.0 confirms the swap fix before any recompile.

    python verify_rope_fix.py --config model_config/llama8B_block0_p64_attn.json
"""
import os
import json
import argparse
import numpy as np

THETA = 500000.0


def rope_perm_idx(D):
    half = D // 2
    p = np.empty(D, dtype=np.int64)
    p[0::2] = np.arange(half) + half
    p[1::2] = np.arange(half)
    return p


def cos_(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="model_config/llama8B_block0_p64_attn.json")
    ap.add_argument("--dump-dir", default="dump_intermediates")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    P, dim = cfg["P"], cfg["dim"]
    n_heads, n_kv, head_dim = cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]
    pos = cfg["prefill_len"]

    dim_p_pe = dim // P
    kv_dim = n_kv * head_dim
    kv_dim_p_pe = kv_dim // P
    gqa = n_heads // n_kv
    pes_p_kv_head = P // n_kv
    _dim_p_pe = dim_p_pe if dim_p_pe % 2 == 0 else dim_p_pe - 1
    _kv_dim_p_pe = kv_dim_p_pe if kv_dim_p_pe % 2 == 0 else kv_dim_p_pe - 1
    kperm = rope_perm_idx(head_dim)

    # Host freq layout — identical to launch_verify.py (per-PE, segment s = px % pes_p_kv_head,
    # tiled gqa_group_size times across the group-inner Q blocks).
    inv = THETA ** (-2.0 * np.arange(head_dim // 2) / head_dim)
    base_cos = np.cos(pos * inv).astype(np.float16)
    base_sin = np.sin(pos * inv).astype(np.float16)
    half_per_s = kv_dim_p_pe // 2
    fc = np.zeros((P, _dim_p_pe // 2), np.float16)
    fs = np.zeros((P, _dim_p_pe // 2), np.float16)
    for px in range(P):
        s = px % pes_p_kv_head
        fc[px] = np.tile(base_cos[s * half_per_s:(s + 1) * half_per_s], gqa)
        fs[px] = np.tile(base_sin[s * half_per_s:(s + 1) * half_per_s], gqa)

    grid = np.load(os.path.join(args.dump_dir, "raw_QKV_post_reduce.npy"))   # [P,P,96] pre-rope
    ref_Qr = np.load(os.path.join(args.dump_dir, "ref_Qr.npy"))
    ref_Kr = np.load(os.path.join(args.dump_dir, "ref_Kr.npy"))

    def rope_pe(buf, fcpx, fspx, npairs, fixed):
        out = buf.copy()
        for i in range(npairs):
            e, o = out[2 * i], out[2 * i + 1]
            c, s = float(fcpx[i]), float(fspx[i])
            if fixed:                       # HF-correct for even=x2, odd=x1
                out[2 * i]     = e * c + o * s
                out[2 * i + 1] = o * c - e * s
            else:                           # current kernel (slots swapped)
                out[2 * i]     = o * c - e * s
                out[2 * i + 1] = e * c + o * s
        return out

    def reconstruct(which, fixed):
        if which == "Q":
            width, off, npairs, n = dim_p_pe, 0, _dim_p_pe // 2, n_heads
        else:
            width, off, npairs, n = kv_dim_p_pe, dim_p_pe, _kv_dim_p_pe // 2, n_kv
        perm = np.zeros(P * width, np.float64)
        for px in range(P):
            buf = grid[0, px, off:off + width].astype(np.float64)
            perm[px * width:(px + 1) * width] = rope_pe(buf, fc[px], fs[px], npairs, fixed)
        if which == "Q":                    # invert Option-C column permutation
            kern = np.zeros(dim, np.float64)
            for h in range(n_heads):
                kvh, g = h // gqa, h % gqa
                for s in range(pes_p_kv_head):
                    old = h * head_dim + s * kv_dim_p_pe
                    new = kvh * pes_p_kv_head * dim_p_pe + s * dim_p_pe + g * kv_dim_p_pe
                    kern[old:old + kv_dim_p_pe] = perm[new:new + kv_dim_p_pe]
        else:
            kern = perm
        v = kern.reshape(n, head_dim)
        hf = np.zeros_like(v); hf[:, kperm] = v
        return hf

    print(f"config P={P} pos={pos}   dump={args.dump_dir}")
    print("  (device observed: QKV_tile Q cos=-0.006190, K cos=-0.017265 — CURRENT below should match)\n")
    for which, ref in (("Q", ref_Qr), ("K", ref_Kr)):
        cur = cos_(reconstruct(which, False), ref)
        fix = cos_(reconstruct(which, True), ref)
        print(f"  {which} post-rope vs HF:   CURRENT cos={cur:+.6f}    FIXED cos={fix:+.6f}")
    print("\n=> FIXED ~1.0 & CURRENT ~0  => kernel rope output slots are swapped; "
          "the swap in xq_rope/xk_rope is the fix.")


if __name__ == "__main__":
    main()
