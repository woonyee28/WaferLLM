import json
import os
import struct
import argparse
import numpy as np

from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view, calculate_cycles
from cerebras.sdk.debug.debug_util import debug_util
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder

# ─────────────────────────────────────────────────────────────────────────────
# Reference implementations matching decode.csl exactly
# ─────────────────────────────────────────────────────────────────────────────

EPS = 1e-6   # matches CSL: const eps: f16 = 0.000001

def rmsnorm_csl(X, W, norm_dim):
    """
    Faithful RMSNorm — matches the hardened decode.csl rmsnorm_x:
      ss[b]     = sum_d X[b,d]^2            (Y-reduced over the full model dim)
      X_norm[b] = X[b] * W / sqrt(ss[b] / norm_dim + eps)
    Numerator is X (not X^2), divisor is the full model dim (not head_dim).
    """
    x32 = X.astype(np.float32)
    w32 = W.astype(np.float32)
    ss = np.sum(x32 ** 2, axis=-1, keepdims=True)
    return (x32 * w32 / np.sqrt(ss / norm_dim + EPS)).astype(np.float16)

def rope_csl(x, freqs_cos, freqs_sin):
    """
    CSL formula (decode.csl xq_rope / xk_rope):
      even_new = x_odd * cos - x_even * sin
      odd_new  = x_even * cos + x_odd * sin
    """
    x32 = x.astype(np.float32)
    cos = freqs_cos.astype(np.float32)
    sin = freqs_sin.astype(np.float32)
    xe = x32[:, 0::2]
    xo = x32[:, 1::2]
    out = np.empty_like(x32)
    out[:, 0::2] = xo * cos - xe * sin
    out[:, 1::2] = xe * cos + xo * sin
    return out.astype(np.float16)

def fast_exp_csl(x):
    """CSL fast_exp: (1 + x/256)^4  — matches decode.csl exactly (2 squarings, f16).
    NOT e^x: only a very crude approximation. Valid only near x≈0."""
    tmp = x.astype(np.float16)
    tmp = np.float16(1.0) + tmp / np.float16(256.0)
    tmp = tmp * tmp   # ^2
    tmp = tmp * tmp   # ^4
    return tmp

def softmax_csl(score):
    """CSL softmax_score: fast_exp approx with Y-reduced max and sum."""
    s32 = score.astype(np.float32)
    mx = np.max(s32, axis=-1, keepdims=True)
    e = fast_exp_csl(s32 - mx)
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float16)

def compute_reference(X, W_norm, W_Q_perm, W_K, W_V, freqs_cos, freqs_sin,
                      XKCache, XVCache, P, n_heads, n_kv_heads, head_dim,
                      dim_p_pe, kv_dim_p_pe, max_seq_len_p_pe, prefill_len_p_pe,
                      gqa_group_size, pes_p_kv_head, total_steps=1):
    """
    Phase 3 reference for the final decode step after total_steps steps.
    XKCache: [bsz, kv_dim, max_seq_len] — per-batch, only cols 0..prefill_len-1 valid.
    XVCache: [bsz, max_seq_len, kv_dim] — per-batch, only rows 0..prefill_len-1 valid.
    Interleaved layout: XKCache[b, :, t] is K for batch b, global token t.
    After total_steps decode steps: each step s writes to PE py=s%P at slot
      prefill_len_p_pe + s//P.  Since X is constant across steps, K_new and V_new
    are identical for every decode step (but differ per batch element).
    Returns dict with iter_num_per_pe, attn_per_head, output_grid, etc.
    """
    kv_dim = n_kv_heads * head_dim
    prefill_len = P * prefill_len_p_pe
    alpha = np.float16(1.0 / np.sqrt(head_dim))
    bsz = X.shape[0]

    # Step 1: RMSNorm (faithful — numerator X, divisor = full model dim = P*dim_p_pe)
    X_norm = rmsnorm_csl(X, W_norm, P * dim_p_pe)

    # Step 2: Projections using permuted W_Q and compact W_K/V
    Q_perm = (X_norm.astype(np.float32) @ W_Q_perm.astype(np.float32)).astype(np.float16)  # [bsz, dim]
    K_new = (X_norm.astype(np.float32) @ W_K.astype(np.float32)).astype(np.float16)         # [bsz, kv_dim]
    V_new = (X_norm.astype(np.float32) @ W_V.astype(np.float32)).astype(np.float16)         # [bsz, kv_dim]

    # Step 3: RoPE — Q rotates over the full dim, K over the first kv_dim//2 freqs.
    Q_rope = rope_csl(Q_perm, freqs_cos, freqs_sin)
    K_rope = rope_csl(K_new, freqs_cos[:kv_dim // 2], freqs_sin[:kv_dim // 2])

    # After total_steps decode steps:
    # PE py receives writes at decode steps {py, py+P, py+2P, ...}.
    # iter_num_per_pe[py] = prefill_len_p_pe + number of those steps < total_steps.
    iter_num_per_pe = np.array(
        [prefill_len_p_pe + len(range(py, total_steps, P)) for py in range(P)], dtype=int
    )

    # Build extended K/V cache with all total_steps decode tokens — PER BATCH.
    # Since X is constant across steps, every decode step produces the same K_new/V_new per batch.
    # Decode token at step s lands at global index prefill_len + s.
    seq_len = prefill_len + total_steps
    K_ext = np.zeros((bsz, kv_dim, seq_len), dtype=np.float32)
    K_ext[:, :, :prefill_len] = XKCache[:, :, :prefill_len].astype(np.float32)
    for s in range(total_steps):
        for b in range(bsz):
            K_ext[b, :, prefill_len + s] = K_rope[b, :].astype(np.float32)

    V_ext = np.zeros((bsz, seq_len, kv_dim), dtype=np.float32)
    V_ext[:, :prefill_len, :] = XVCache[:, :prefill_len, :].astype(np.float32)
    for s in range(total_steps):
        for b in range(bsz):
            V_ext[b, prefill_len + s, :] = V_new[b, :].astype(np.float32)

    # Step 4+5: Attention per Q-head over seq_len tokens — per batch
    score_per_head = np.zeros((n_heads, bsz, seq_len), dtype=np.float32)
    for h in range(n_heads):
        kv_head = h // gqa_group_size
        g = h % gqa_group_size
        Q_h = np.zeros((bsz, head_dim), dtype=np.float32)
        for s in range(pes_p_kv_head):
            px = kv_head * pes_p_kv_head + s
            col_start = px * dim_p_pe + g * kv_dim_p_pe
            Q_h[:, s * kv_dim_p_pe:(s + 1) * kv_dim_p_pe] = Q_rope[:, col_start:col_start + kv_dim_p_pe].astype(np.float32)
        for b in range(bsz):
            K_kv_b = K_ext[b, kv_head * head_dim:(kv_head + 1) * head_dim, :]  # [head_dim, seq_len]
            score_per_head[h, b, :] = Q_h[b:b+1, :] @ K_kv_b

    score_scaled = (score_per_head * float(alpha)).astype(np.float16)
    attn_per_head = np.stack([softmax_csl(score_scaled[h]) for h in range(n_heads)])  # [n_heads, bsz, seq_len]

    # Step 6: Output per head — BATCH-OUTER layout [bsz, gqa_group_size, kv_dim_p_pe]
    # output[b * dim_p_pe + g * kv_dim_p_pe : ... + kv_dim_p_pe] = attn[h,b] @ V[b]
    output_grid = np.zeros((P, P, bsz * dim_p_pe), dtype=np.float32)
    for px in range(P):
        kv_head = px // pes_p_kv_head
        for g in range(gqa_group_size):
            h = kv_head * gqa_group_size + g
            for b in range(bsz):
                v_slice_b = V_ext[b, :, px * kv_dim_p_pe:(px + 1) * kv_dim_p_pe]  # [seq_len, kv_dim_p_pe]
                out_g = (attn_per_head[h, b:b+1, :].astype(np.float32) @ v_slice_b.astype(np.float32)).astype(np.float16)  # [1, kv_dim_p_pe]
                start = b * dim_p_pe + g * kv_dim_p_pe
                for py in range(P):
                    output_grid[py, px, start:start + kv_dim_p_pe] = out_g[0, :]

    return {
        'X_norm':           X_norm,
        'Q_perm':           Q_perm,
        'K_new':            K_new,
        'V_new':            V_new,
        'K_ext':            K_ext.astype(np.float16),
        'V_ext':            V_ext.astype(np.float16),
        'iter_num_per_pe':  iter_num_per_pe,
        'attn_per_head':    attn_per_head,        # [n_heads, bsz, seq_len]
        'output_grid':      output_grid.astype(np.float16),  # [P, P, bsz*dim_p_pe]
    }

# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def sep(title):
    print(f"\n{'═'*64}")
    print(f"  {title}")
    print(f"{'═'*64}")

def show(name, arr, n=6):
    flat = arr.astype(np.float32).ravel()
    vals = "  ".join(f"{v:8.4f}" for v in flat[:n])
    tail = " ..." if len(flat) > n else ""
    print(f"  {name:36s} {vals}{tail}")

def show_all(name, grid, n=None, ref_grid=None, row_label="py"):
    """Print a grid [rows, cols, data] in WSE-3 debug format."""
    n_rows, n_cols = grid.shape[0], grid.shape[1]
    print(f"  {name}:")
    for row in range(n_rows):
        sim_row = np.concatenate([grid[row, c, :] for c in range(n_cols)]).astype(np.float32)
        truncated = n is not None and len(sim_row) > n
        def fmt(arr):
            v = "  ".join(f"{v:8.4f}" for v in (arr[:n] if truncated else arr))
            return v + (" ..." if truncated else "")
        prefix = f"    {row_label}={row}"
        if ref_grid is None:
            print(f"{prefix}: [{fmt(sim_row)}]")
        else:
            ref_row = np.concatenate(
                [ref_grid[row, c, :] for c in range(n_cols)]
            ).astype(np.float32)
            print(f"{prefix} sim: [{fmt(sim_row)}]")
            print(f"{prefix} ref: [{fmt(ref_row)}]")

def cmp(name, sim, ref, atol=0.15):
    s = sim.astype(np.float32).ravel()
    r = ref.astype(np.float32).ravel()
    max_err = float(np.max(np.abs(s - r)))
    ok = max_err <= atol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:42s}  max_err={max_err:.5f}")
    if not ok:
        print(f"         sim: {s[:6]}")
        print(f"         ref: {r[:6]}")
    return ok

def reconstruct_QKV(qkv_grid, bsz, dim_p_pe, kv_dim_p_pe, P):
    """
    qkv_grid: [P, P, bsz*(dim_p_pe + 2*kv_dim_p_pe)]  (indexed [py, px, ...])
    Returns Q [bsz, dim], K [bsz, kv_dim], V [bsz, kv_dim] using py=0.
    """
    dim = P * dim_p_pe
    kv_dim = P * kv_dim_p_pe
    Q = np.zeros((bsz, dim), dtype=np.float16)
    K = np.zeros((bsz, kv_dim), dtype=np.float16)
    V = np.zeros((bsz, kv_dim), dtype=np.float16)
    for px in range(P):
        pe = qkv_grid[0, px, :]  # [bsz*(dim_p_pe + 2*kv_dim_p_pe)]
        Q[:, px * dim_p_pe:(px + 1) * dim_p_pe] = pe[0:bsz * dim_p_pe].reshape(bsz, dim_p_pe)
        K[:, px * kv_dim_p_pe:(px + 1) * kv_dim_p_pe] = pe[bsz * dim_p_pe:bsz * (dim_p_pe + kv_dim_p_pe)].reshape(bsz, kv_dim_p_pe)
        V[:, px * kv_dim_p_pe:(px + 1) * kv_dim_p_pe] = pe[bsz * (dim_p_pe + kv_dim_p_pe):].reshape(bsz, kv_dim_p_pe)
    return Q, K, V

def reconstruct_score(score_grid, bsz, max_seq_len_p_pe, P, n_heads, n_kv_heads, gqa_group_size, iter_num_per_pe=None):
    """
    Phase 3: score_grid [P, P, bsz*gqa_group_size*max_seq_len_p_pe].
    Score layout: BATCH-OUTER [bsz, gqa_group_size, iter_num].
    iter_num_per_pe: array of length P giving valid token count per PE-y.
    Returns attn [n_heads, bsz, total_tokens] assembled from per-PE valid slots.
    Total tokens = sum(iter_num_per_pe).
    Global token t → PE py=t%P, slot t//P.
    """
    if iter_num_per_pe is None:
        iter_num_per_pe = np.full(P, max_seq_len_p_pe, dtype=int)
    pes_p_kv_head = P // n_kv_heads
    total_tokens = int(np.sum(iter_num_per_pe))
    attn = np.zeros((n_heads, bsz, total_tokens), dtype=np.float16)
    for h in range(n_heads):
        kv_head = h // gqa_group_size
        g = h % gqa_group_size
        px_kv = kv_head * pes_p_kv_head   # representative PE for KV-head
        for py in range(P):
            iters = iter_num_per_pe[py]
            pe = score_grid[py, px_kv, :]  # [bsz * gqa_group_size * max_seq_len_p_pe]
            for slot in range(iters):
                t_global = py + slot * P
                if t_global < total_tokens:
                    for b in range(bsz):
                        idx = b * gqa_group_size * iters + g * iters + slot  # batch-outer
                        attn[h, b, t_global] = pe[idx]
    return attn

def reconstruct_output(out_grid, bsz, dim_p_pe, P):
    """
    out_grid: [P, P, bsz*dim_p_pe]
    Returns output [bsz, dim] using py=0 (Y-reduce → all py identical).
    """
    out = np.zeros((bsz, P * dim_p_pe), dtype=np.float16)
    for px in range(P):
        out[:, px * dim_p_pe: (px + 1) * dim_p_pe] = out_grid[0, px, :].reshape(bsz, dim_p_pe)
    return out

def d2h(runner, sym_id, P, bsz, data_per_pe, io_dtype, memcpy_order):
    """Read P×P PEs, each contributing data_per_pe f16 values."""
    buf = np.zeros(P * bsz * data_per_pe * P, dtype=np.uint32)
    runner.memcpy_d2h(
        buf, sym_id, 0, 0, P, P, bsz * data_per_pe,
        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    return memcpy_view(buf, np.dtype(np.float16)).reshape(P, P, bsz * data_per_pe)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 interleaved KV cache tiling
# ─────────────────────────────────────────────────────────────────────────────

def tile_kcache_interleaved(K_cache, P, bsz, kv_dim_p_pe, max_seq_len_p_pe, prefill_len_p_pe):
    """
    Build KCache_tile[P, P, bsz * kv_dim_p_pe * max_seq_len_p_pe] for H2D ROW_MAJOR.
    K_cache: [bsz, kv_dim, max_seq_len], interleaved so K_cache[b, :, t] = K for batch b, global token t.
    On-chip layout per PE: [bsz, kv_dim_p_pe, max_seq_len_p_pe] — bsz slowest.
    tile[py, px, b*kv_dim_p_pe*max_seq_len_p_pe + k*max_seq_len_p_pe + s]
        = K_cache[b, px*kv_dim_p_pe+k, py+s*P].
    """
    cache_per_pe = kv_dim_p_pe * max_seq_len_p_pe
    tile = np.zeros((P, P, bsz * cache_per_pe), dtype=np.float16)
    for b in range(bsz):
        for py in range(P):
            for px in range(P):
                for s in range(prefill_len_p_pe):
                    token = py + s * P
                    k_row_start = px * kv_dim_p_pe
                    k_row_end = (px + 1) * kv_dim_p_pe
                    for k in range(kv_dim_p_pe):
                        tile[py, px, b * cache_per_pe + k * max_seq_len_p_pe + s] = \
                            K_cache[b, k_row_start + k, token]
    return tile

def tile_vcache_interleaved(V_cache, P, bsz, kv_dim_p_pe, max_seq_len_p_pe, prefill_len_p_pe):
    """
    Build VCache_tile[P, P, bsz * max_seq_len_p_pe * kv_dim_p_pe] for H2D ROW_MAJOR.
    V_cache: [bsz, max_seq_len, kv_dim], interleaved so V_cache[b, t, :] = V for batch b, global token t.
    On-chip layout per PE: [bsz, max_seq_len_p_pe, kv_dim_p_pe] — bsz slowest.
    tile[py, px, b*max_seq_len_p_pe*kv_dim_p_pe + s*kv_dim_p_pe + j]
        = V_cache[b, py+s*P, px*kv_dim_p_pe+j].
    """
    cache_per_pe = max_seq_len_p_pe * kv_dim_p_pe
    tile = np.zeros((P, P, bsz * cache_per_pe), dtype=np.float16)
    for b in range(bsz):
        for py in range(P):
            for px in range(P):
                j_start = px * kv_dim_p_pe
                j_end = (px + 1) * kv_dim_p_pe
                for s in range(prefill_len_p_pe):
                    token = py + s * P
                    tile[py, px, b * cache_per_pe + s * kv_dim_p_pe:(b * cache_per_pe + (s + 1) * kv_dim_p_pe)] = \
                        V_cache[b, token, j_start:j_end]
    return tile

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])

def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)

class Config:
    def __init__(self):
        self.P = 8
        self.bsz = 1
        self.group_num = 2
        self.dim = 64
        self.n_heads = 1
        self.n_kv_heads = 1
        self.head_dim = 64
        self.max_seq_len = 128
        self.prefill_len = 64
        self.ffn_dim = 64

def parse_args():
    parser = argparse.ArgumentParser(description="WSE-3-GQA decode simulator")
    parser.add_argument("--config", default="config.json", type=str)
    parser.add_argument("--simple", action="store_true",
                        help="Use constant matrices (fill value) for easy hand-verification")
    parser.add_argument("--fill", type=float, default=1.0,
                        help="Constant fill value for --simple mode (default 1.0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--validate", action="store_true",
                        help="Read intermediate buffers and compare against numpy reference")
    parser.add_argument("--steps", type=int, default=1,
                        help="Total decode steps to run (default 1). Each step appends one token to the KV cache.")
    parser.add_argument("--cmaddr", type=str, default=None,
                        help="CM address for hardware execution (via SdkLauncher)")
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config()

    if not os.path.exists(args.config):
        print("Host: Use default test values.")
    else:
        with open(args.config) as f:
            config.__dict__.update(json.load(f))

    P            = config.P
    bsz          = config.bsz
    group_num    = config.group_num
    dim          = config.dim
    n_heads      = config.n_heads
    n_kv_heads   = config.n_kv_heads
    head_dim     = config.head_dim
    max_seq_len  = config.max_seq_len
    prefill_len  = config.prefill_len
    ffn_dim      = config.ffn_dim

    dim_p_pe         = dim // P
    pes_p_head       = P // n_heads
    pes_p_kv_head    = P // n_kv_heads
    head_dim_p_pe    = head_dim // P
    max_seq_len_p_pe = max_seq_len // P
    prefill_len_p_pe = prefill_len // P
    ffn_dim_p_pe     = ffn_dim // P
    kv_dim           = n_kv_heads * head_dim
    kv_dim_p_pe      = kv_dim // P
    gqa_group_size   = n_heads // n_kv_heads
    _kv_dim_p_pe     = (kv_dim_p_pe // 2) * 2

    print(f"Host: P={P}  bsz={bsz}  dim={dim}  n_heads={n_heads}  n_kv_heads={n_kv_heads}  gqa_group_size={gqa_group_size}")
    print(f"      head_dim={head_dim}  max_seq_len={max_seq_len}  prefill_len={prefill_len}  ffn_dim={ffn_dim}")
    print(f"      dim_p_pe={dim_p_pe}  kv_dim_p_pe={kv_dim_p_pe}  pes_p_head={pes_p_head}  pes_p_kv_head={pes_p_kv_head}")
    print(f"      max_seq_len_p_pe={max_seq_len_p_pe}  prefill_len_p_pe={prefill_len_p_pe}")
    if args.simple:
        print(f"  [simple mode: fill={args.fill}  identity RoPE  easy hand-verification]")

    if args.seed is not None:
        np.random.seed(args.seed)

    io_dtype    = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    # ─── Build input / weight tensors ─────────────────────────────────────────
    def mk(*shape, fill=args.fill):
        if args.simple:
            return np.full(shape, fill, dtype=np.float16)
        return np.random.rand(*shape).astype(np.float16)

    def mk_dim(*shape):
        if args.simple:
            X = np.zeros(shape, dtype=np.float16)
            for idx in np.ndindex(*shape):
                X[idx] = sum(idx) * 0.1
            return X
        return np.random.rand(*shape).astype(np.float16)

    X_raw    = mk(1, bsz * dim, fill=.1)
    for j in range(bsz):
        for i in range(dim):
            X_raw[0, j * dim + i] =0.1*( j + 1) # j + 1
            #X[0, i*dim_p_pe*bsz + j*dim_p_pe : i*dim_p_pe*bsz + (j+1)*dim_p_pe] = i + 1 # j + 1
    print(f"X_raw: {X_raw}")
    W_raw    = mk(1, dim, fill=args.fill)

    tensor_q_weight  =mk(dim, dim, fill=.1)
    tensor_k_weight  = mk(dim, kv_dim, fill=.2)
    tensor_v_weight  = mk(dim, kv_dim, fill=.3)

    W_norm_flat = W_raw.ravel()  # [dim]

    _dim_p_pe = dim_p_pe if (dim_p_pe % 2 == 0) else dim_p_pe - 1

    # RoPE freqs
    if args.simple:
        base_cos = np.ones(head_dim // 2, dtype=np.float16)
        base_sin = np.zeros(head_dim // 2, dtype=np.float16)
    else:
        base_cos = np.random.rand(head_dim // 2).astype(np.float16)
        base_sin = np.random.rand(head_dim // 2).astype(np.float16)

    _half_head = (head_dim_p_pe // 2 * 2) // 2
    pe_freqs_sin = np.zeros((P, _dim_p_pe // 2), dtype=np.float16)
    pe_freqs_cos = np.zeros((P, _dim_p_pe // 2), dtype=np.float16)
    for _px in range(P):
        s     = _px % pes_p_head
        start = s * _half_head
        end   = start + _half_head
        if _half_head > 0:
            pe_freqs_sin[_px, :] = np.tile(base_sin[start:end], n_heads)
            pe_freqs_cos[_px, :] = np.tile(base_cos[start:end], n_heads)
    tensor_freqs_sin = np.tile(pe_freqs_sin.ravel(), (P, 1))
    tensor_freqs_cos = np.tile(pe_freqs_cos.ravel(), (P, 1))

    freqs_cos_ref = np.zeros(dim // 2, dtype=np.float16)
    freqs_sin_ref = np.zeros(dim // 2, dtype=np.float16)
    for _px in range(P):
        off = _px * (dim_p_pe // 2)
        freqs_cos_ref[off: off + _dim_p_pe // 2] = pe_freqs_cos[_px, :]
        freqs_sin_ref[off: off + _dim_p_pe // 2] = pe_freqs_sin[_px, :]

    # ─── Phase 3 KV cache: max_seq_len capacity, only prefill_len positions filled ─
    # Batched: tensor_XKCache[bsz, kv_dim, max_seq_len], tensor_XVCache[bsz, max_seq_len, kv_dim].
    # Interleaved layout: tensor_XKCache[b, :, t] = K for batch b, global token t.
    # Only tokens 0..prefill_len-1 are pre-loaded; rest are zeros.
    tensor_XKCache = np.zeros((bsz, kv_dim, max_seq_len), dtype=np.float16)
    tensor_XVCache = np.zeros((bsz, max_seq_len, kv_dim), dtype=np.float16)
    if args.simple:
        tensor_XKCache[:, :, :prefill_len] = np.full((bsz, kv_dim, prefill_len), 0.04, dtype=np.float16)
        tensor_XVCache[:, :prefill_len, :] = np.full((bsz, prefill_len, kv_dim), 0.05, dtype=np.float16)
    else:
        tensor_XKCache[:, :, :prefill_len] = np.random.rand(bsz, kv_dim, prefill_len).astype(np.float16)
        tensor_XVCache[:, :prefill_len, :] = np.random.rand(bsz, prefill_len, kv_dim).astype(np.float16)

    tensor_o_weight    = mk(dim, dim)
    tensor_up_weight   = mk(dim, ffn_dim)
    tensor_gate_weight = mk(dim, ffn_dim)
    tensor_down_weight = mk(ffn_dim, dim)

    # ─── W_Q / W_O offline column permutation (Option C GQA) ──────────────────
    W_Q_perm = np.zeros((dim, dim), dtype=np.float16)
    W_O_perm = np.zeros((dim, dim), dtype=np.float16)
    for _h in range(n_heads):
        _kv_head = _h // gqa_group_size
        _g = _h % gqa_group_size
        for _s in range(pes_p_kv_head):
            _old_col = _h * head_dim + _s * kv_dim_p_pe
            _new_col = _kv_head * pes_p_kv_head * dim_p_pe + _s * dim_p_pe + _g * kv_dim_p_pe
            W_Q_perm[:, _new_col:_new_col + kv_dim_p_pe] = tensor_q_weight[:, _old_col:_old_col + kv_dim_p_pe]
            W_O_perm[_new_col:_new_col + kv_dim_p_pe, :] = tensor_o_weight[_old_col:_old_col + kv_dim_p_pe, :]

    # ─── Tile inputs for the PE grid ──────────────────────────────────────────
    # For bsz > 1: each PE row py must hold [X[0,py_slice], X[1,py_slice], ...].
    # Correct: reshape to [bsz, P, dim_p_pe], transpose to [P, bsz, dim_p_pe], then flatten batch+feat.
    X_per_pe = X_raw.reshape(bsz, P, dim_p_pe).transpose(1, 0, 2).reshape(P, bsz * dim_p_pe)
    tensor_X = np.tile(X_per_pe, reps=(1, P))
    tensor_W = np.tile(W_raw.reshape(P, dim_p_pe), reps=(1, P))

    def tile_weight_row(W, rows, cols):
        """W: [rows*P, cols*P] → PE tile [P, P, rows*cols]"""
        r = W.reshape(P, rows, P, cols).transpose(0, 2, 1, 3).reshape(P, P, rows * cols)
        return r

    Q_tile  = tile_weight_row(W_Q_perm,           dim_p_pe, dim_p_pe)
    K_tile  = tile_weight_row(tensor_k_weight,    dim_p_pe, kv_dim_p_pe)
    V_tile  = tile_weight_row(tensor_v_weight,    dim_p_pe, kv_dim_p_pe)
    O_tile  = tile_weight_row(W_O_perm,           dim_p_pe, dim_p_pe)
    UP_tile = tile_weight_row(tensor_up_weight,  dim_p_pe, ffn_dim_p_pe)
    GT_tile = tile_weight_row(tensor_gate_weight, dim_p_pe, ffn_dim_p_pe)
    DN_tile = tile_weight_row(tensor_down_weight, ffn_dim_p_pe, dim_p_pe)

    # Phase 3 interleaved KV cache tiling (batched):
    # PE(px, py) gets K/V for tokens {py, py+P, ..., py+(prefill_len_p_pe-1)*P}, per batch.
    # On-chip layout: [bsz, kv_dim_p_pe, max_seq_len_p_pe] and [bsz, max_seq_len_p_pe, kv_dim_p_pe].
    KCache_tile = tile_kcache_interleaved(tensor_XKCache, P, bsz, kv_dim_p_pe, max_seq_len_p_pe, prefill_len_p_pe)
    VCache_tile = tile_vcache_interleaved(tensor_XVCache, P, bsz, kv_dim_p_pe, max_seq_len_p_pe, prefill_len_p_pe)

    # ─── Runner ───────────────────────────────────────────────────────────────
    if args.cmaddr:
        runner = SdkRuntime("out", cmaddr=args.cmaddr)
    else:
        runner = SdkRuntime("out", simfab_numthreads=64, msg_level='INFO')
    runner.load()
    runner.run()

    sym_X           = runner.get_id("X")
    sym_W           = runner.get_id("W")
    sym_Q_weight    = runner.get_id("Q_weight")
    sym_K_weight    = runner.get_id("K_weight")
    sym_V_weight    = runner.get_id("V_weight")
    sym_freqs_sin   = runner.get_id("freqs_sin")
    sym_freqs_cos   = runner.get_id("freqs_cos")
    sym_XKCache     = runner.get_id("XKCache")
    sym_XVCache     = runner.get_id("XVCache")
    sym_O_weight    = runner.get_id("O_weight")
    sym_UP_weight   = runner.get_id("UP_weight")
    sym_GATE_weight = runner.get_id("GATE_weight")
    sym_DOWN_weight = runner.get_id("DOWN_weight")
    sym_timer_buf   = runner.get_id("timer_buf")
    sym_time_ref    = runner.get_id("time_ref")
    sym_debug       = runner.get_id("debug")

    sym_freqs_cos_sym = runner.get_id("freqs_cos")
    sym_freqs_sin_sym = runner.get_id("freqs_sin")

    if args.validate:
        sym_QKV_post_proj    = runner.get_id("QKV_post_proj")
        sym_QKV_post_reduce  = runner.get_id("QKV_post_reduce")
        sym_QKV_tile         = runner.get_id("QKV_tile")
        sym_score_post_gemv  = runner.get_id("score_post_gemv")
        sym_score_post_reduce = runner.get_id("score_post_reduce")
        sym_score            = runner.get_id("score")
        sym_output_tile      = runner.get_id("output_tile")
        sym_local_sum_dbg    = runner.get_id("local_sum_dbg")

    # ─── H2D memcpy ───────────────────────────────────────────────────────────
    def h2d(sym, flat_arr, count_per_pe):
        u32 = input_array_to_u32(flat_arr.ravel(), 1, 1)
        runner.memcpy_h2d(
            sym, u32, 0, 0, P, P, count_per_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

    h2d(sym_X,    tensor_X,            bsz * dim_p_pe)
    h2d(sym_W,    tensor_W,            dim_p_pe)
    h2d(sym_Q_weight,    Q_tile,        dim_p_pe * dim_p_pe)
    h2d(sym_K_weight,    K_tile,        dim_p_pe * kv_dim_p_pe)
    h2d(sym_V_weight,    V_tile,        dim_p_pe * kv_dim_p_pe)
    h2d(sym_freqs_sin,   tensor_freqs_sin, _dim_p_pe // 2)
    h2d(sym_freqs_cos,   tensor_freqs_cos, _dim_p_pe // 2)
    h2d(sym_XKCache,     KCache_tile,   bsz * kv_dim_p_pe * max_seq_len_p_pe)
    h2d(sym_XVCache,     VCache_tile,   bsz * max_seq_len_p_pe * kv_dim_p_pe)
    h2d(sym_O_weight,    O_tile,        dim_p_pe * dim_p_pe)
    h2d(sym_UP_weight,   UP_tile,       dim_p_pe * ffn_dim_p_pe)
    h2d(sym_GATE_weight, GT_tile,       dim_p_pe * ffn_dim_p_pe)
    h2d(sym_DOWN_weight, DN_tile,       ffn_dim_p_pe * dim_p_pe)

    # ─── Launch ───────────────────────────────────────────────────────────────
    runner.launch("init_task", nonblock=False)

    repeat_steps = args.steps
    warmup_steps = 0
    runner.launch("decode_host", np.int16(repeat_steps), np.int16(warmup_steps), nonblock=False)

    # ─── D2H: debug output ────────────────────────────────────────────────────
    debug_buf = np.zeros(P * bsz * dim, dtype=np.uint32)
    runner.memcpy_d2h(
        debug_buf, sym_debug, 0, 0, P, P, bsz * dim_p_pe,
        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    debug_out = memcpy_view(debug_buf, np.dtype(np.float16)).reshape(P, bsz * dim)

    # ─── D2H: freqs readback (always, for debugging RoPE) ────────────────────
    freqs_cos_grid = d2h(runner, sym_freqs_cos_sym, P, 1, _dim_p_pe // 2, io_dtype, memcpy_order)
    freqs_sin_grid = d2h(runner, sym_freqs_sin_sym, P, 1, _dim_p_pe // 2, io_dtype, memcpy_order)

    # ─── D2H: intermediate validation buffers ─────────────────────────────────
    if args.validate:
        proj_grid         = d2h(runner, sym_QKV_post_proj,    P, bsz, dim_p_pe + 2 * kv_dim_p_pe,  io_dtype, memcpy_order)
        reduce_grid       = d2h(runner, sym_QKV_post_reduce,  P, bsz, dim_p_pe + 2 * kv_dim_p_pe,  io_dtype, memcpy_order)
        qkv_grid          = d2h(runner, sym_QKV_tile,         P, bsz, dim_p_pe + 2 * kv_dim_p_pe,  io_dtype, memcpy_order)
        # Score buffers: all gqa_group_size groups, max_seq_len_p_pe per group (only iter_num[py] valid)
        score_gemv_grid   = d2h(runner, sym_score_post_gemv,  P, bsz, gqa_group_size * max_seq_len_p_pe, io_dtype, memcpy_order)
        score_reduce_grid = d2h(runner, sym_score_post_reduce, P, bsz, gqa_group_size * max_seq_len_p_pe, io_dtype, memcpy_order)
        score_grid        = d2h(runner, sym_score,             P, bsz, gqa_group_size * max_seq_len_p_pe,  io_dtype, memcpy_order)
        out_grid          = d2h(runner, sym_output_tile,       P, bsz, dim_p_pe,      io_dtype, memcpy_order)

        # ── DBG: rmsnorm_x post-reduce sum-of-squares per batch (localize the bsz>1 halving) ──
        ls_grid  = d2h(runner, sym_local_sum_dbg, P, 1, bsz, io_dtype, memcpy_order)   # [P, P, bsz]
        ls_expect = np.sum(X_raw.reshape(bsz, dim).astype(np.float32) ** 2, axis=1) / P  # per-PE partial (pre-reduce)
        sep("DBG — rmsnorm_x PRE-reduce per-PE local_sum partial  (should equal sum_d X[b,d]^2 / P)")
        print(f"  expected per batch : {[round(float(v), 5) for v in ls_expect]}")
        for _py in range(min(P, 2)):
            for _px in range(min(P, 2)):
                got = ls_grid[_py, _px, :].astype(np.float32)
                ratio = [round(float(got[b] / ls_expect[b]), 4) if ls_expect[b] != 0 else 0.0 for b in range(bsz)]
                print(f"  PE(py={_py},px={_px})  local_sum={[round(float(v),5) for v in got]}   sim/expected={ratio}")

    # ─── D2H: timer ───────────────────────────────────────────────────────────
    timer_buf_1d = np.zeros(P * P * 3, dtype=np.uint32)
    runner.memcpy_d2h(
        timer_buf_1d, sym_timer_buf, 0, 0, P, P, 3,
        streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR, nonblock=False
    )
    timer_hwl = timer_buf_1d.view(np.float32).reshape((P, P, 3))

    runner.stop()

    # ─── Timing ───────────────────────────────────────────────────────────────
    cycles = np.zeros((P, P))
    for pe_x in range(P):
        for pe_y in range(P):
            cycles[pe_y, pe_x] = calculate_cycles(timer_hwl[pe_y, pe_x, :])
    print(f"\nHost: mean cycles = {cycles.mean() / repeat_steps:.0f}")

    # ─── Validation ───────────────────────────────────────────────────────────
    if not args.validate:
        print("\nDebug output (first row):", debug_out[0, :8], "...")
        return

    # Reconstruct full-rank tensors from the PE grid
    X_flat   = X_raw.reshape(bsz, dim)
    W_norm_v = W_norm_flat                             # [dim]

    ref = compute_reference(
        X_flat, W_norm_v,
        W_Q_perm, tensor_k_weight, tensor_v_weight,
        freqs_cos_ref, freqs_sin_ref,
        tensor_XKCache, tensor_XVCache,
        P, n_heads, n_kv_heads, head_dim,
        dim_p_pe, kv_dim_p_pe, max_seq_len_p_pe, prefill_len_p_pe,
        gqa_group_size, pes_p_kv_head,
        total_steps=repeat_steps
    )

    iter_num_per_pe = ref['iter_num_per_pe']
    prefill_len_ref = P * prefill_len_p_pe
    seq_len_ref     = prefill_len_ref + repeat_steps  # total tokens after all steps

    all_ok = True

    # ── Freqs sanity check ─────────────────────────────────────────────────────
    sep("Freqs sanity check  (verify PE received correct freqs_cos/sin)")
    for px in range(min(P, 4)):
        sim_cos = freqs_cos_grid[0, px, :]
        exp_cos = pe_freqs_cos[px, :]
        sim_sin = freqs_sin_grid[0, px, :]
        exp_sin = pe_freqs_sin[px, :]
        cos_ok = float(np.max(np.abs(sim_cos.astype(np.float32) - exp_cos.astype(np.float32)))) < 0.01
        sin_ok = float(np.max(np.abs(sim_sin.astype(np.float32) - exp_sin.astype(np.float32)))) < 0.01
        print(f"  PE(px={px})  sim_cos={sim_cos.tolist()}  exp_cos={exp_cos.tolist()}  {'OK' if cos_ok else 'MISMATCH'}")
        print(f"          sim_sin={sim_sin.tolist()}  exp_sin={exp_sin.tolist()}  {'OK' if sin_ok else 'MISMATCH'}")
    print()

    # ── Step 2a: Post-projection (local partial GEMV, before Y-reduce) ────────
    sep("Step 2a — Post-projection  (local partial GEMV per PE, before Y-reduce)")
    X_norm = ref['X_norm']
    proj_ref = np.zeros((P, P, bsz * (dim_p_pe + 2 * kv_dim_p_pe)), dtype=np.float32)
    for py in range(P):
        x_row = X_norm[:, py*dim_p_pe:(py+1)*dim_p_pe].astype(np.float32)
        for px in range(P):
            q_p = x_row @ W_Q_perm[py*dim_p_pe:(py+1)*dim_p_pe, px*dim_p_pe:(px+1)*dim_p_pe].astype(np.float32)
            k_p = x_row @ tensor_k_weight[py*dim_p_pe:(py+1)*dim_p_pe, px*kv_dim_p_pe:(px+1)*kv_dim_p_pe].astype(np.float32)
            v_p = x_row @ tensor_v_weight[py*dim_p_pe:(py+1)*dim_p_pe, px*kv_dim_p_pe:(px+1)*kv_dim_p_pe].astype(np.float32)
            proj_ref[py, px, :bsz*dim_p_pe] = q_p.ravel()
            proj_ref[py, px, bsz*dim_p_pe:bsz*(dim_p_pe+kv_dim_p_pe)] = k_p.ravel()
            proj_ref[py, px, bsz*(dim_p_pe+kv_dim_p_pe):] = v_p.ravel()
    proj_ref = proj_ref.astype(np.float16)
    show_all("proj_sim", proj_grid)
    show_all("proj_ref", proj_ref)
    all_ok &= cmp("Step2a Q-partial  sim vs ref", proj_grid[:,:,:bsz*dim_p_pe], proj_ref[:,:,:bsz*dim_p_pe])
    all_ok &= cmp("Step2a K-partial  sim vs ref", proj_grid[:,:,bsz*dim_p_pe:bsz*(dim_p_pe+kv_dim_p_pe)], proj_ref[:,:,bsz*dim_p_pe:bsz*(dim_p_pe+kv_dim_p_pe)])
    all_ok &= cmp("Step2a V-partial  sim vs ref", proj_grid[:,:,bsz*(dim_p_pe+kv_dim_p_pe):], proj_ref[:,:,bsz*(dim_p_pe+kv_dim_p_pe):])

    # ── Step 2b: Post-Y-reduce (full projection, before RoPE) ────────────────
    sep("Step 2b — Post-Y-reduce  (full Q/K/V, all py identical, before RoPE)")
    Q_full = (X_norm.astype(np.float32) @ W_Q_perm.astype(np.float32)).astype(np.float16)
    K_full = (X_norm.astype(np.float32) @ tensor_k_weight.astype(np.float32)).astype(np.float16)
    V_full = (X_norm.astype(np.float32) @ tensor_v_weight.astype(np.float32)).astype(np.float16)
    reduce_ref = np.zeros((P, P, bsz * (dim_p_pe + 2 * kv_dim_p_pe)), dtype=np.float16)
    for px in range(P):
        for py in range(P):
            reduce_ref[py, px, :bsz*dim_p_pe] = Q_full[:, px*dim_p_pe:(px+1)*dim_p_pe].ravel()
            reduce_ref[py, px, bsz*dim_p_pe:bsz*(dim_p_pe+kv_dim_p_pe)] = K_full[:, px*kv_dim_p_pe:(px+1)*kv_dim_p_pe].ravel()
            reduce_ref[py, px, bsz*(dim_p_pe+kv_dim_p_pe):] = V_full[:, px*kv_dim_p_pe:(px+1)*kv_dim_p_pe].ravel()
    show_all("reduce_sim", reduce_grid)
    show_all("reduce_ref", reduce_ref)
    all_ok &= cmp("Step2b Q-reduced  sim vs ref", reduce_grid[:,:,:bsz*dim_p_pe], reduce_ref[:,:,:bsz*dim_p_pe])
    all_ok &= cmp("Step2b K-reduced  sim vs ref", reduce_grid[:,:,bsz*dim_p_pe:bsz*(dim_p_pe+kv_dim_p_pe)], reduce_ref[:,:,bsz*dim_p_pe:bsz*(dim_p_pe+kv_dim_p_pe)])
    all_ok &= cmp("Step2b V-reduced  sim vs ref", reduce_grid[:,:,bsz*(dim_p_pe+kv_dim_p_pe):], reduce_ref[:,:,bsz*(dim_p_pe+kv_dim_p_pe):])

    # Direct kernel-rope check: rope the known-correct pre-rope tensors (reduce_grid, which
    # PASSED Step 2b) with numpy, per-PE, and compare to the kernel's post-rope qkv_grid.
    # Isolates the kernel rope op from the score/cache/permutation (which Step 5 confounds).
    sep("Step 2c — Post-RoPE  (kernel qkv_grid vs numpy rope of pre-rope reduce_grid)")
    _kv_dp = kv_dim_p_pe - (kv_dim_p_pe % 2)   # even part (whole pairs)
    def _rope_vec(v, cos, sin):
        v = v.astype(np.float32); out = v.copy()
        e = v[0::2]; o = v[1::2]
        out[0::2] = o * cos - e * sin
        out[1::2] = e * cos + o * sin
        return out
    qkv_rope_ref = reduce_grid.astype(np.float32).copy()
    for py in range(P):
        for px in range(P):
            cos = pe_freqs_cos[px].astype(np.float32)
            sin = pe_freqs_sin[px].astype(np.float32)
            for b in range(bsz):
                qs = b * dim_p_pe
                qkv_rope_ref[py, px, qs:qs + _dim_p_pe] = _rope_vec(
                    reduce_grid[py, px, qs:qs + _dim_p_pe], cos[:_dim_p_pe // 2], sin[:_dim_p_pe // 2])
                ks = bsz * dim_p_pe + b * kv_dim_p_pe
                qkv_rope_ref[py, px, ks:ks + _kv_dp] = _rope_vec(
                    reduce_grid[py, px, ks:ks + _kv_dp], cos[:_kv_dp // 2], sin[:_kv_dp // 2])
    all_ok &= cmp("Step2c Q-roped  sim vs ref", qkv_grid[:, :, :bsz*dim_p_pe], qkv_rope_ref[:, :, :bsz*dim_p_pe])
    all_ok &= cmp("Step2c K-roped  sim vs ref",
                  qkv_grid[:, :, bsz*dim_p_pe:bsz*(dim_p_pe+kv_dim_p_pe)],
                  qkv_rope_ref[:, :, bsz*dim_p_pe:bsz*(dim_p_pe+kv_dim_p_pe)])

    if args.simple:
        xnorm = float(ref['X_norm'][0, 0])
        q_exp = dim * xnorm
        print(f"\n  Hand-check: X_norm≈{xnorm:.4f}  Q[any]=dim×X_norm×W_q={dim}×{xnorm:.4f}×{args.fill}={q_exp:.4f}")

    Q_perm_full = (ref['X_norm'].astype(np.float32) @ W_Q_perm.astype(np.float32)).astype(np.float16)
    # RoPE Q and the new-token K to match the kernel (Step 2c confirms the kernel ropes them).
    # Prefill K in the cache is seeded (un-roped) in both kernel and ref -> only K_new is roped.
    Q_rope_full = rope_csl(Q_perm_full, freqs_cos_ref, freqs_sin_ref)
    K_new_roped = rope_csl(ref['K_new'], freqs_cos_ref[:kv_dim // 2], freqs_sin_ref[:kv_dim // 2])

    # ── Build per-PE reference for step 5a and 5b (Phase 3: variable iter_num, all groups) ──
    # Layout mirrors device: BATCH-OUTER [bsz, gqa_group_size, iter_num].
    #   base(b,g) = b * gqa_group_size * iters + g * iters.
    # K_pe(px, py): slot s → token = py + s*P; prefill from tensor_XKCache,
    #   decode slots [prefill_len_p_pe..iters-1] all hold K_new (X constant across steps).
    alpha_val = np.float16(1.0 / np.sqrt(head_dim))

    # Score layout: BATCH-OUTER [bsz, gqa_group_size, iter_num]
    #   base(b,g) = b * gqa_group_size * iters + g * iters
    score_5a_ref = np.zeros((P, P, bsz * gqa_group_size * max_seq_len_p_pe), dtype=np.float16)
    for py in range(P):
        iters = iter_num_per_pe[py]
        for px in range(P):
            for b in range(bsz):
                K_pe = np.zeros((kv_dim_p_pe, iters), dtype=np.float32)
                for s in range(min(iters, prefill_len_p_pe)):
                    token = py + s * P
                    K_pe[:, s] = tensor_XKCache[b, px * kv_dim_p_pe:(px + 1) * kv_dim_p_pe, token].astype(np.float32)
                # Decode slots: all have same K_new since X is constant across steps
                for j in range(prefill_len_p_pe, iters):
                    K_pe[:, j] = K_new_roped[b, px * kv_dim_p_pe:(px + 1) * kv_dim_p_pe].astype(np.float32)
                for g in range(gqa_group_size):
                    q_g = Q_rope_full[b:b+1, px * dim_p_pe + g * kv_dim_p_pe : px * dim_p_pe + (g + 1) * kv_dim_p_pe].astype(np.float32)
                    partial = (q_g @ K_pe).astype(np.float16)  # [1, iters]
                    base = b * gqa_group_size * iters + g * iters  # batch-outer layout
                    score_5a_ref[py, px, base : base + iters] = partial[0, :]

    # score_5b_ref: after KV-head-scoped X-reduce + alpha scale (all groups, batch-outer)
    score_5b_ref = np.zeros((P, P, bsz * gqa_group_size * max_seq_len_p_pe), dtype=np.float16)
    for py in range(P):
        iters = iter_num_per_pe[py]
        for kv_head_i in range(n_kv_heads):
            for g in range(gqa_group_size):
                kv_sum = np.zeros((bsz, iters), dtype=np.float32)
                for px in range(kv_head_i * pes_p_kv_head, (kv_head_i + 1) * pes_p_kv_head):
                    for b in range(bsz):
                        base = b * gqa_group_size * iters + g * iters  # batch-outer
                        kv_sum[b, :] += score_5a_ref[py, px, base : base + iters].astype(np.float32)
                kv_sum_scaled = (kv_sum * float(alpha_val)).astype(np.float16)
                for px in range(kv_head_i * pes_p_kv_head, (kv_head_i + 1) * pes_p_kv_head):
                    for b in range(bsz):
                        base = b * gqa_group_size * iters + g * iters  # batch-outer
                        score_5b_ref[py, px, base : base + iters] = kv_sum_scaled[b, :]

    show_all("qkv_grid", qkv_grid)

    # ── Step 5a: post-GEMV (per-PE partial, before KV-head-scoped reduce) ────────
    sep(f"Step 5a — post-score GEMV  (all {gqa_group_size} groups per PE, before KV-head-scoped X-reduce)")
    print(f"  Phase 3: iter_num per PE = {iter_num_per_pe.tolist()}  (prefill_len_p_pe={prefill_len_p_pe} + {repeat_steps} decode step(s))")
    show_all("score_gemv_sim", score_gemv_grid)
    show_all("score_5a_ref",   score_5a_ref)
    for kv_head_i in range(n_kv_heads):
        px0 = kv_head_i * pes_p_kv_head
        for g in range(gqa_group_size):
            for py in range(P):
                iters = iter_num_per_pe[py]
                for b in range(bsz):
                    base = b * gqa_group_size * iters + g * iters  # batch-outer
                    all_ok &= cmp(
                        f"Step5a score_post_gemv kv_head={kv_head_i} g={g} py={py} b={b} (px={px0})",
                        score_gemv_grid[py, px0, base : base + iters],
                        score_5a_ref[py, px0, base : base + iters]
                    )

    # ── Step 5b: post-KV-head-scoped-reduce + scale (before softmax) ──────────
    sep(f"Step 5b — post KV-head-scoped X-reduce + alpha scale  (all {gqa_group_size} groups, before softmax)")
    show_all("score_reduce_sim", score_reduce_grid)
    show_all("score_5b_ref",     score_5b_ref)
    for kv_head_i in range(n_kv_heads):
        px0 = kv_head_i * pes_p_kv_head
        for g in range(gqa_group_size):
            for py in range(P):
                iters = iter_num_per_pe[py]
                for b in range(bsz):
                    base = b * gqa_group_size * iters + g * iters  # batch-outer
                    all_ok &= cmp(
                        f"Step5b score_post_reduce kv_head={kv_head_i} g={g} py={py} b={b}",
                        score_reduce_grid[py, px0, base : base + iters],
                        score_5b_ref[py, px0, base : base + iters]
                    )
        # Verify all PEs in KV-head block are identical (for each py, all groups)
        for py in range(P):
            iters = iter_num_per_pe[py]
            for px in range(px0 + 1, (kv_head_i + 1) * pes_p_kv_head):
                max_diff = float(np.max(np.abs(
                    score_reduce_grid[py, px, :gqa_group_size * iters * bsz].astype(np.float32)
                    - score_reduce_grid[py, px0, :gqa_group_size * iters * bsz].astype(np.float32)
                )))
                print(f"  [{'OK' if max_diff < 0.01 else 'MISMATCH'}] kv_head {kv_head_i} py={py}: px={px} identical to px={px0}  max_diff={max_diff:.5f}")

    # ── Step 5c: post-softmax (attn weights) ──────────────────────────────────
    sep("Step 5c — post-softmax  (final attention weights)")
    # Build attn_ref_grid in PE-grid format [P, P, bsz*gqa_group_size*max_seq_len_p_pe]
    # iter_num-packed layout — matches score_grid exactly (no padding, no reordering)
    attn_ref_grid = np.zeros((P, P, bsz * gqa_group_size * max_seq_len_p_pe), dtype=np.float16)
    for px in range(P):
        kv_head = px // pes_p_kv_head
        for py in range(P):
            iters = iter_num_per_pe[py]
            for g in range(gqa_group_size):
                h = kv_head * gqa_group_size + g
                for b in range(bsz):
                    for slot in range(iters):
                        t_global = py + slot * P
                        if t_global < seq_len_ref:
                            attn_ref_grid[py, px, b * gqa_group_size * iters + g * iters + slot] = ref['attn_per_head'][h, b, t_global]

    show_all("attn_sim", score_grid)
    show_all("attn_ref", attn_ref_grid)

    # Compare raw buffers (iter_num-packed, same format as device) — same pattern as 5a/5b
    for kv_head_i in range(n_kv_heads):
        px0 = kv_head_i * pes_p_kv_head
        for g in range(gqa_group_size):
            for py in range(P):
                iters = iter_num_per_pe[py]
                for b in range(bsz):
                    base = b * gqa_group_size * iters + g * iters  # batch-outer
                    all_ok &= cmp(
                        f"Step5c attn kv_head={kv_head_i} g={g} py={py} b={b}",
                        score_grid[py, px0, base : base + iters],
                        attn_ref_grid[py, px0, base : base + iters]
                    )
        # Sum ≈ 1.0 check per head (summed over all PEs' valid slots from raw buffer)
        for g in range(gqa_group_size):
            h = kv_head_i * gqa_group_size + g
            px0 = kv_head_i * pes_p_kv_head
            for b in range(bsz):
                s_sim = sum(
                    float(np.sum(score_grid[py, px0,
                        b * gqa_group_size * iter_num_per_pe[py] + g * iter_num_per_pe[py] :
                        b * gqa_group_size * iter_num_per_pe[py] + g * iter_num_per_pe[py] + iter_num_per_pe[py]
                    ].astype(np.float32)))
                    for py in range(P)
                )
                s_ref = float(np.sum(ref['attn_per_head'][h, b].astype(np.float32)))
                print(f"    sum(attn h={h} b={b}): sim={s_sim:.6f}  ref={s_ref:.6f}  (expect ≈1.0)")

    if args.simple:
        alpha_f = 1.0 / np.sqrt(float(head_dim))
        q_val = float(ref['Q_perm'][0, 0])
        k_val = float(tensor_XKCache[0, 0, 0])
        print(f"\n  Hand-check (g=0): seq_len_ref={seq_len_ref}  alpha={alpha_f:.4f}")

    # ── Step 6: output_matvec_mult ───────────────────────────────────────────
    sep("Step 6 — output_matvec_mult  (gqa_group_size attn@V GEMVs + Y-reduce)")
    show_all("output_sim", out_grid)
    show_all("output_ref", ref['output_grid'])
    all_ok &= cmp("output  sim vs ref", out_grid, ref['output_grid'])

    # ── Summary ───────────────────────────────────────────────────────────────
    sep("Validation Summary")
    print(f"  Config: P={P}  n_heads={n_heads}  n_kv_heads={n_kv_heads}  gqa_group_size={gqa_group_size}")
    print(f"          pes_p_head={pes_p_head}  pes_p_kv_head={pes_p_kv_head}  dim={dim}  kv_dim={kv_dim}")
    print(f"          max_seq_len={max_seq_len}  prefill_len={prefill_len}  seq_len_after_{repeat_steps}_steps={seq_len_ref}")
    print(f"          iter_num_per_pe={iter_num_per_pe.tolist()}")
    print(f"  simple={args.simple}  fill={args.fill}")
    print()
    print(f"  Steps validated (sim = WSE-3 simulator, ref = numpy reference):")
    print(f"    2a.  QKV post-proj       per-PE partial (W_Q_perm + compact K/V)")
    print(f"    2b.  QKV post-Y-reduce   full Q/K/V (before RoPE, which is disabled)")
    print(f"    5a.  Score post-GEMV     g=0 partial per PE, before KV-head-scoped reduce")
    print(f"    5b.  Score post-reduce   g=0 after KV-head-scoped X-reduce + alpha scale")
    print(f"    5c.  Attn weights        n_heads={n_heads} heads, post-softmax, seq_len={seq_len_ref}")
    print(f"    6.   output_matvec       {gqa_group_size} attn@V GEMVs + Y-reduce")
    print()
    print(f"  Overall: {'ALL PASS ✓' if all_ok else 'SOME FAILURES — check output above'}")
    if not all_ok:
        print()
        print("  Tolerance: atol=0.15 (float16 matmul accumulation)")
        print("  If failures are near the tolerance, check for head boundary routing bugs.")

if __name__ == "__main__":
    main()
