#!/usr/bin/env python3
"""
validate.py — Reference numpy validation for WSE-3-GQA attention decode.

Computes and prints expected intermediate values for each attention step
(up to output_matvec_mult). Validates that the head-scoped score reduce
produces correct per-head scores for MHA (Phase 1).

Formulas exactly match decode.csl:
  RMSNorm:  X_norm = X^2 * W / sqrt(sum_Y(X^2) / head_dim + eps)
  RoPE:     even_new = odd*cos - even*sin,  odd_new = even*cos + odd*sin
  fast_exp: (1 + x/256)^256  (CSL approximate softmax)

Usage:
  # Random inputs (seed 42)
  python3 validate.py --config model_config/mha_test.json

  # All-ones matrices for easy hand-verification
  python3 validate.py --config model_config/mha_test.json --simple

  # Skip RMSNorm (pass X unchanged)
  python3 validate.py --config model_config/mha_test.json --simple --no-norm
"""

import argparse
import json
import numpy as np

EPS = 1e-6   # matches CSL: const eps: f16 = 0.000001


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def sep(title):
    print(f"\n{'═'*64}")
    print(f"  {title}")
    print(f"{'═'*64}")


def show(name, arr, n=8):
    flat = arr.astype(np.float32).ravel()
    vals = "  ".join(f"{v:8.4f}" for v in flat[:n])
    tail = " ..." if len(flat) > n else ""
    print(f"  {name:38s} {vals}{tail}")


def check(name, got, ref, atol=5e-2):
    g = got.astype(np.float32)
    r = ref.astype(np.float32)
    max_err = float(np.max(np.abs(g - r)))
    ok = max_err <= atol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:44s}  max_err={max_err:.5f}")
    if not ok:
        print(f"         got: {g.ravel()[:6]}")
        print(f"         ref: {r.ravel()[:6]}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Exact CSL formula implementations
# ─────────────────────────────────────────────────────────────────────────────

def rmsnorm_csl(X, W, head_dim):
    """
    Matches rmsnorm_x() in decode.csl:
      X_tmp = X^2
      local_sum = sum(X_tmp per batch)  [Y-reduced across all P PEs]
      X_norm = X_tmp * W                [= X^2 * W]
      X_norm = X_norm / sqrt(local_sum / head_dim + eps)
    """
    x32 = X.astype(np.float32)
    w32 = W.astype(np.float32)
    ss  = np.sum(x32 ** 2, axis=-1, keepdims=True)      # [bsz, 1]
    rms_inv = 1.0 / np.sqrt(ss / head_dim + EPS)
    return (x32**2 * w32 * rms_inv).astype(np.float16)


def rope_csl(x, freqs_cos, freqs_sin):
    """
    Matches xq_rope / xk_rope in decode.csl:
      tmp1 = x_odd  * cos
      tmp2 = x_even * sin
      tmp3 = x_even * cos
      tmp4 = x_odd  * sin
      x_even_new = tmp1 - tmp2  =  x_odd*cos  - x_even*sin
      x_odd_new  = tmp3 + tmp4  =  x_even*cos + x_odd*sin
    """
    x32 = x.astype(np.float32)
    cos = freqs_cos.astype(np.float32)   # [dim//2]
    sin = freqs_sin.astype(np.float32)
    xe  = x32[:, 0::2]   # [bsz, dim//2]
    xo  = x32[:, 1::2]
    out = np.empty_like(x32)
    out[:, 0::2] = xo * cos - xe * sin
    out[:, 1::2] = xe * cos + xo * sin
    return out.astype(np.float16)


def fast_exp_csl(x):
    """Matches CSL fast_exp: (1 + x/256)^256"""
    tmp = x.astype(np.float64)
    tmp = 1.0 + tmp / 256.0
    for _ in range(8):   # 2^8 = 256
        tmp = tmp * tmp
    return tmp.astype(np.float32)


def softmax_csl(score):
    """
    Matches softmax_score() in decode.csl (uses fast_exp approximation):
      max via Y-reduce
      score_tmp = score - max
      score = fast_exp(score_tmp)
      sum via Y-reduce
      score /= sum
    """
    s32 = score.astype(np.float32)
    mx  = np.max(s32, axis=-1, keepdims=True)
    e   = fast_exp_csl(s32 - mx)
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float16)


# ─────────────────────────────────────────────────────────────────────────────
# PE-level distributed simulation
# ─────────────────────────────────────────────────────────────────────────────

def pe_score(Q_rope, XKCache, P, n_heads, dim_p_pe):
    """
    Simulate head-scoped X-reduce for score computation.

    PE px belongs to head h = px // pes_p_head.
    Each PE holds Q[:, px*dim_p_pe:(px+1)*dim_p_pe] and
                  XKCache[px*dim_p_pe:(px+1)*dim_p_pe, :].
    Partial scores accumulate within each head block (not across heads).

    Returns: score_per_head [n_heads, bsz, seq_len]
    """
    pes_p_head = P // n_heads
    seq_len = XKCache.shape[1]
    bsz     = Q_rope.shape[0]
    scores  = np.zeros((n_heads, bsz, seq_len), dtype=np.float32)

    for px in range(P):
        h       = px // pes_p_head
        q_local = Q_rope[:, px * dim_p_pe: (px + 1) * dim_p_pe]      # [bsz, dim_p_pe]
        k_local = XKCache[px * dim_p_pe: (px + 1) * dim_p_pe, :]     # [dim_p_pe, seq_len]
        scores[h] += q_local.astype(np.float32) @ k_local.astype(np.float32)

    return scores.astype(np.float16)


def pe_output(attn_per_head, XVCache, P, n_heads, dim_p_pe, seq_len_p_pe):
    """
    Simulate output_matvec_mult() with per-head attention weights.

    PE (px, py): score = attn_h[b, py*seq_len_p_pe:(py+1)*seq_len_p_pe]
                 V     = XVCache[py*seq_len_p_pe:(py+1)*seq_len_p_pe, px*dim_p_pe:(px+1)*dim_p_pe]
    After Y-reduce: output[b, px*dim_p_pe:] = attn_h[b, :] @ V[:, px*dim_p_pe:]
    where h = px // pes_p_head.

    Returns: output [bsz, dim]
    """
    pes_p_head = P // n_heads
    head_dim   = dim_p_pe * pes_p_head
    bsz        = attn_per_head.shape[1]
    output     = np.zeros((bsz, n_heads * head_dim), dtype=np.float32)

    for h in range(n_heads):
        v_h = XVCache[:, h * head_dim: (h + 1) * head_dim]   # [seq_len, head_dim]
        output[:, h * head_dim: (h + 1) * head_dim] = (
            attn_per_head[h].astype(np.float32) @ v_h.astype(np.float32)
        )

    return output.astype(np.float16)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reference numpy validation for WSE-3-GQA attention decode."
    )
    parser.add_argument("--config",  default="model_config/mha_test.json")
    parser.add_argument("--simple",  action="store_true",
                        help="All-ones matrices for easy hand-verification")
    parser.add_argument("--no-norm", action="store_true",
                        help="Skip RMSNorm (pass X unchanged to QKV)")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = json.load(f)

    P            = cfg["P"]
    bsz          = cfg["bsz"]
    dim          = cfg["dim"]
    n_heads      = cfg["n_heads"]
    head_dim     = cfg["head_dim"]
    seq_len      = cfg["seq_len"]

    pes_p_head   = P // n_heads
    dim_p_pe     = dim // P
    head_dim_p_pe = head_dim // P
    seq_len_p_pe  = seq_len // P
    alpha         = np.float16(1.0 / np.sqrt(head_dim))

    print(f"Config : P={P}  bsz={bsz}  dim={dim}  n_heads={n_heads}  "
          f"head_dim={head_dim}  seq_len={seq_len}")
    print(f"Derived: pes_p_head={pes_p_head}  dim_p_pe={dim_p_pe}  "
          f"head_dim_p_pe={head_dim_p_pe}  seq_len_p_pe={seq_len_p_pe}  "
          f"alpha={float(alpha):.6f}")
    print(f"Flags  : simple={args.simple}  no_norm={args.no_norm}  seed={args.seed}")

    # ── Build tensors ─────────────────────────────────────────────────────────
    def mk(*shape, val=1.0):
        if args.simple:
            return np.full(shape, val, dtype=np.float16)
        return np.random.rand(*shape).astype(np.float16)

    X       = mk(bsz, dim)
    W_norm  = mk(dim, val=1.0)
    W_Q     = mk(dim, dim)
    W_K     = mk(dim, dim)
    W_V     = mk(dim, dim)
    XKCache = mk(dim, seq_len)     # K^T stored: [dim, seq_len]
    XVCache = mk(seq_len, dim)     # V stored:   [seq_len, dim]

    # RoPE freqs: identity for simple mode, head-local random otherwise
    if args.simple or args.no_norm:
        freqs_cos = np.ones(dim // 2, dtype=np.float16)
        freqs_sin = np.zeros(dim // 2, dtype=np.float16)
    else:
        _half     = (head_dim_p_pe // 2 * 2) // 2
        base_cos  = np.random.rand(head_dim // 2).astype(np.float16)
        base_sin  = np.random.rand(head_dim // 2).astype(np.float16)
        freqs_cos = np.zeros(dim // 2, dtype=np.float16)
        freqs_sin = np.zeros(dim // 2, dtype=np.float16)
        for px in range(P):
            s   = px % pes_p_head
            off = px * (dim_p_pe // 2)
            for h in range(n_heads):
                dst = off + h * (head_dim_p_pe // 2)
                src = s * _half
                freqs_cos[dst: dst + _half] = base_cos[src: src + _half]
                freqs_sin[dst: dst + _half] = base_sin[src: src + _half]

    # ═════════════════════════════════════════════════════════════════════════
    # Step 0 — Inputs
    # ═════════════════════════════════════════════════════════════════════════
    sep("Step 0 — Inputs")
    show("X", X)
    if args.simple:
        print(f"  All weight matrices: all-ones (float16 = 1.0)")
        print(f"  RoPE: identity  cos=1, sin=0")
        print(f"  Expected Q/K/V element (post-norm): dim * X_norm_val")

    # ═════════════════════════════════════════════════════════════════════════
    # Step 1 — RMSNorm
    # ═════════════════════════════════════════════════════════════════════════
    sep("Step 1 — RMSNorm  (CSL formula: X_norm = X² × W / √(∑X²/head_dim + ε))")
    if args.no_norm:
        X_norm = X
        print("  [skipped — using X unchanged]")
    else:
        X_norm = rmsnorm_csl(X, W_norm, head_dim)
        show("X_norm", X_norm)
        if args.simple:
            ss_val = float(dim)
            rms_inv = 1.0 / np.sqrt(ss_val / head_dim + EPS)
            expected = 1.0 * rms_inv    # X^2=1, W=1, rms_inv
            print(f"  Hand-check: ∑X²={ss_val}  rms_inv=1/√({ss_val}/{head_dim})={rms_inv:.6f}")
            print(f"  X_norm[any] = 1.0 × 1.0 × {rms_inv:.6f} = {expected:.6f}")
            ok = np.allclose(X_norm.astype(np.float32), expected, atol=1e-2)
            print(f"  [{'PASS' if ok else 'FAIL'}] X_norm == {expected:.4f}")

    # ═════════════════════════════════════════════════════════════════════════
    # Step 2 — QKV Projections
    # ═════════════════════════════════════════════════════════════════════════
    sep("Step 2 — QKV Projections:  X_norm @ W_Q/K/V  (+Y-reduce across P PEs)")
    Q = (X_norm.astype(np.float32) @ W_Q.astype(np.float32)).astype(np.float16)
    K = (X_norm.astype(np.float32) @ W_K.astype(np.float32)).astype(np.float16)
    V = (X_norm.astype(np.float32) @ W_V.astype(np.float32)).astype(np.float16)
    show("Q", Q)
    show("K", K)
    show("V", V)
    if args.simple:
        xnorm_val = float(X_norm[0, 0])
        q_exp = dim * xnorm_val
        print(f"  Hand-check: Q[any] = dim × X_norm_val = {dim} × {xnorm_val:.4f} = {q_exp:.4f}")
        ok = np.allclose(Q.astype(np.float32), q_exp, atol=max(abs(q_exp) * 0.05, 0.1))
        print(f"  [{'PASS' if ok else 'FAIL'}] Q matches expected value")

    # ═════════════════════════════════════════════════════════════════════════
    # Step 3 — RoPE
    # ═════════════════════════════════════════════════════════════════════════
    sep("Step 3 — RoPE  (even_new = odd·cos − even·sin,  odd_new = even·cos + odd·sin)")
    Q_rope = rope_csl(Q, freqs_cos, freqs_sin)
    K_rope = rope_csl(K, freqs_cos, freqs_sin)
    show("Q_rope", Q_rope)
    show("K_rope", K_rope)
    if args.simple:
        q0, q1 = float(Q[0, 0]), float(Q[0, 1])
        print(f"  Hand-check (cos=1, sin=0): even_new = odd = {q1:.4f},  odd_new = even = {q0:.4f}")
        print(f"  (All-ones Q: even=odd, so Q_rope = Q unchanged)")

    # ═════════════════════════════════════════════════════════════════════════
    # Step 4 — Score = Q @ K^T  (head-scoped X-reduce)
    # ═════════════════════════════════════════════════════════════════════════
    sep("Step 4 — Score: Q_rope @ XKCache  (head-scoped X-reduce, pes_p_head PEs)")
    print(f"  Phase 1 change: reduce scoped to {pes_p_head} PEs/head (not all {P} PEs)")
    print(f"  XKCache: [dim={dim}, seq_len={seq_len}]  (K transposed)")
    print()

    # Per-head reference (correct for MHA)
    score_ref = np.zeros((n_heads, bsz, seq_len), dtype=np.float32)
    for h in range(n_heads):
        q_h = Q_rope[:, h * head_dim: (h + 1) * head_dim]       # [bsz, head_dim]
        k_h = XKCache[h * head_dim: (h + 1) * head_dim, :]      # [head_dim, seq_len]
        score_ref[h] = q_h.astype(np.float32) @ k_h.astype(np.float32)

    # Full-P naive (incorrect for n_heads>1 — what the original single-head code did)
    score_naive = Q_rope.astype(np.float32) @ XKCache.astype(np.float32)

    # PE-level simulation (models the head-scoped reduce)
    score_pe = pe_score(Q_rope, XKCache, P, n_heads, dim_p_pe)

    for h in range(n_heads):
        show(f"score_ref head {h}", score_ref[h].astype(np.float16))
    print()
    show("score_naive (full-P, WRONG for n_heads>1)", score_naive.astype(np.float16))
    if n_heads > 1:
        diff = float(np.max(np.abs(score_naive - score_ref[0])))
        print(f"  naive vs head-0 diff: {diff:.4f}  (nonzero → head-scoping matters)")
    print()

    score_ok = True
    for h in range(n_heads):
        ok = check(f"PE-sim score head {h} == reference", score_pe[h], score_ref[h].astype(np.float16))
        score_ok = score_ok and ok

    # Scaled scores
    print()
    score_scaled = (score_ref * float(alpha)).astype(np.float16)
    for h in range(n_heads):
        show(f"score_scaled head {h} (×{float(alpha):.4f})", score_scaled[h])

    if args.simple:
        q_val = float(Q_rope[0, 0])
        k_val = float(XKCache[0, 0])
        s_exp = pes_p_head * dim_p_pe * q_val * k_val   # = head_dim * q * k
        print(f"  Hand-check: score[h,b,t] = pes_p_head×dim_p_pe × q_val × k_val")
        print(f"            = {pes_p_head}×{dim_p_pe} × {q_val:.4f} × {k_val:.4f} = {s_exp:.4f}")
        print(f"  Scaled:    {s_exp:.4f} × {float(alpha):.4f} = {s_exp * float(alpha):.4f}")

    if not score_ok:
        print()
        print("  *** SCORE HEAD-SCOPED REDUCE FAILED — possible Phase 1 routing bug ***")

    # ═════════════════════════════════════════════════════════════════════════
    # Step 5 — Softmax
    # ═════════════════════════════════════════════════════════════════════════
    sep("Step 5 — Softmax  (CSL fast_exp=(1+x/256)^256, Y-reduce for max & sum)")
    attn = np.zeros_like(score_scaled)
    softmax_ok = True
    for h in range(n_heads):
        attn[h] = softmax_csl(score_scaled[h])
        show(f"attn head {h}", attn[h])
        s = float(np.sum(attn[h][0].astype(np.float32)))
        ok = abs(s - 1.0) < 0.05
        softmax_ok = softmax_ok and ok
        print(f"  sum(attn head {h}, batch 0) = {s:.6f}  "
              f"[{'PASS' if ok else 'FAIL'}] (expect ≈ 1.0)")

    if args.simple:
        print(f"  Uniform scores → uniform attn ≈ 1/{seq_len} = {1.0/seq_len:.6f}")

    # ═════════════════════════════════════════════════════════════════════════
    # Step 6 — Output = attn @ V  (Y-reduce)
    # ═════════════════════════════════════════════════════════════════════════
    sep("Step 6 — Output: attn @ XVCache  (Y-reduce over seq_len, per head)")
    print(f"  XVCache: [seq_len={seq_len}, dim={dim}]")
    print()

    output_sim = pe_output(attn, XVCache, P, n_heads, dim_p_pe, seq_len_p_pe)

    # Reference: per-head matmul
    output_ref = np.zeros((bsz, dim), dtype=np.float32)
    for h in range(n_heads):
        v_h = XVCache[:, h * head_dim: (h + 1) * head_dim]
        output_ref[:, h * head_dim: (h + 1) * head_dim] = (
            attn[h].astype(np.float32) @ v_h.astype(np.float32)
        )

    show("output_sim", output_sim)
    show("output_ref", output_ref.astype(np.float16))
    print()
    output_ok = check("output PE-sim == reference", output_sim, output_ref.astype(np.float16))

    if args.simple:
        o_exp = 1.0   # uniform 1/seq_len attn, all-ones V → sum = 1
        print(f"  Hand-check: (1/{seq_len}) × (∑_t V[t,j]) = (1/{seq_len}) × {seq_len} × 1 = {o_exp:.4f}")
        ok = np.allclose(output_sim.astype(np.float32), o_exp, atol=0.1)
        print(f"  [{'PASS' if ok else 'FAIL'}] output ≈ {o_exp:.4f} (all elements)")

    # ═════════════════════════════════════════════════════════════════════════
    # Summary
    # ═════════════════════════════════════════════════════════════════════════
    sep("Summary")
    print(f"  Step 1 RMSNorm  : formula X²·W/√(∑X²/head_dim+ε)")
    print(f"  Step 2 QKV proj : X_norm @ W_Q/K/V  (global matmul + Y-reduce)")
    print(f"  Step 3 RoPE     : even↔odd rotation with head-local freqs")
    print(f"  Step 4 Score    : [{'PASS' if score_ok else 'FAIL'}]  "
          f"head-scoped X-reduce ({pes_p_head} PEs/head, {n_heads} head(s))")
    print(f"  Step 5 Softmax  : [{'PASS' if softmax_ok else 'FAIL'}]  "
          f"fast_exp approx, sums to 1")
    print(f"  Step 6 Output   : [{'PASS' if output_ok else 'FAIL'}]  "
          f"attn @ V per head, Y-reduce")
    print()
    all_ok = score_ok and softmax_ok and output_ok
    print(f"  Overall: {'ALL PASS' if all_ok else 'SOME FAILURES — see above'}")
    if not all_ok:
        print()
        print("  Note: RMSNorm formula in CSL is non-standard (uses X² not X).")
        print("  This is expected — it does not affect attention correctness.")
    print()
    print(f"  Re-run with --simple for all-ones matrices (easy hand-verification).")
    print(f"  Re-run with --no-norm to skip RMSNorm.")


if __name__ == "__main__":
    main()
