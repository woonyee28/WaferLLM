# wyn: validating attention test on the SIMULATOR (local, no cluster).
#
# Purpose: reproduce the P=128 attention regime (head_dim_p_pe==1, seq_len_p_pe==1)
# at P=8 with a numpy oracle, to check whether the head-sequential GQA attention
# (head-major QKVO sharding + matmul_T score + causal mask) is numerically correct.
#
# RoPE is disabled by sending cos=1, sin=0. The kernel's rope then just swaps each
# adjacent pair; because Q and K get the SAME swap, the score dot-product is
# unchanged, and V is not roped — so the oracle is plain causal GQA attention.
#
# We read back Z_mid (post-attention residual = X + attn) and score_dbg (raw head-0
# Q.K^T, snapshotted before alpha/mask) and compare to numpy.
#
# Run:  python compile.py --mode sim --config model_config/test-sim-attn-p8.json
#       cs_python launch_sim_attn.py --config model_config/test-sim-attn-p8.json

import numpy as np
import argparse
import os
import json

from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder


def untile_flat_1d(input_flat_1d, P, seq_len_p_pe, dim_p_pe):
    a = input_flat_1d.reshape(P, P, dim_p_pe, seq_len_p_pe)
    a = a.transpose(0, 3, 1, 2)
    return a.reshape(seq_len_p_pe * P, dim_p_pe * P)


def reduce_root_map(P):
    # Mirrors matmul_T_reduce_add() in prefill.csl: co -> physical column (reduce_root).
    root = P // 2
    f = np.zeros(P, dtype=int)
    for co in range(P):
        if co == 0:
            f[co] = 0
        elif co <= root:
            f[co] = 2 * co - 1
        else:
            f[co] = 2 * (P - co)
    return f  # f[co] = column that co deposits into


def assignId(pc, P):
    send_id = 0
    recv_id = 0
    pc = pc + 1
    if pc % 2 == 0:
        send_id = pc - 2
        recv_id = pc + 2
    else:
        send_id = pc + 2
        recv_id = pc - 2
    if pc == 1:
        send_id = 3
        recv_id = 2
    if pc == 2:
        send_id = 1
        recv_id = min(recv_id, P)
    if P % 2 == 0:
        if pc == P - 1:
            send_id = P
            recv_id = P - 3
        if pc == P:
            send_id = P - 2
            recv_id = P - 1
    else:
        if pc == P - 1:
            send_id = max(send_id, 1)
            recv_id = P
        if pc == P:
            send_id = P - 1
            recv_id = P - 2
    return send_id - 1, recv_id - 1


class Config:
    def __init__(self):
        self.P = 8
        self.dim = 32
        self.n_heads = 4
        self.n_kv_heads = 2
        self.head_dim = 8
        self.seq_len = 8
        self.ffn_dim = 32


def parse_args():
    parser = argparse.ArgumentParser(description="Attention validation on simulator")
    parser.add_argument("--config", default="config.json", type=str)
    args = parser.parse_args()
    return args


def report(name, got, ref):
    g = got.astype(np.float64).ravel()
    r = ref.astype(np.float64).ravel()
    max_abs = np.max(np.abs(g - r))
    mean_abs = np.mean(np.abs(g - r))
    denom = np.linalg.norm(g) * np.linalg.norm(r)
    cos = float(g @ r / denom) if denom > 0 else 0.0
    verdict = "PASS" if cos >= 0.999 else "FAIL"
    print(f"[{name}] cos={cos:.6f}  max_abs={max_abs:.4e}  mean_abs={mean_abs:.4e}  -> {verdict}")
    return cos


def main():
    args = parse_args()
    config = Config()
    if os.path.exists(args.config):
        with open(args.config) as f:
            config.__dict__.update(json.load(f))

    P = config.P
    dim = config.dim
    seq_len = config.seq_len
    ffn_dim = config.ffn_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim

    dim_p_pe = dim // P
    seq_len_p_pe = seq_len // P
    ffn_dim_p_pe = ffn_dim // P
    head_dim_p_pe = head_dim // P
    kv_dim = n_kv_heads * head_dim
    kv_dim_p_pe = kv_dim // P
    group = n_heads // n_kv_heads

    _dim_p_pe = dim_p_pe if (dim_p_pe % 2 == 0) else dim_p_pe - 1
    alpha = 1.0 / np.sqrt(head_dim)
    eps = 1e-5

    print(f"P={P} dim={dim} n_heads={n_heads} n_kv_heads={n_kv_heads} head_dim={head_dim} "
          f"seq_len={seq_len} | dim_p_pe={dim_p_pe} head_dim_p_pe={head_dim_p_pe} "
          f"seq_len_p_pe={seq_len_p_pe} group={group}")

    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    # ----- logical inputs (modest scale to stay in fp16 range) -----
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((seq_len, dim)) * 0.5).astype(np.float16)
    Wq = (rng.standard_normal((dim, n_heads * head_dim)) * 0.1).astype(np.float16)
    Wk = (rng.standard_normal((dim, kv_dim)) * 0.1).astype(np.float16)
    Wv = (rng.standard_normal((dim, kv_dim)) * 0.1).astype(np.float16)
    Wo = (rng.standard_normal((n_heads * head_dim, dim)) * 0.1).astype(np.float16)
    # FFN weights only need to keep the kernel from NaNing; they don't affect Z_mid.
    Wup = (rng.standard_normal((dim, ffn_dim)) * 0.1).astype(np.float16)
    Wgate = (rng.standard_normal((dim, ffn_dim)) * 0.1).astype(np.float16)
    Wdown = (rng.standard_normal((ffn_dim, dim)) * 0.1).astype(np.float16)

    # RMSNorm weight = ONES for this run, so the (py-vs-px) W-tiling layout can't
    # affect the result -> isolates the attention mechanics + head sharding + mask.
    W = np.ones((1, dim), dtype=np.float16)
    tensor_W = np.tile(W.reshape(P, dim_p_pe), reps=(1, P))
    W2 = np.ones((1, dim), dtype=np.float16)
    tensor_W2 = np.tile(W2.reshape(P, dim_p_pe), reps=(1, P))

    # ----- ind permutation (same as launch_sim / launch_device) -----
    ind = np.zeros((P, P)).astype(int)
    for i in range(P):
        for j in range(P):
            if i == 0:
                ind[0, j] = j
            elif i == 1:
                _, ind[1, j] = assignId(ind[0, j], P)
            else:
                if (i - 1) % 2 == 0:
                    _, ind[i, j] = assignId(ind[i - 2, j], P)
                else:
                    ind[i, j], _ = assignId(ind[i - 2, j], P)

    # ----- head-major sharding (copied from launch_device.py) -----
    def shard_qkv_headmajor(Wmat, n_h):   # Wmat: [dim, n_h*head_dim]
        out = np.zeros((P, P, n_h * dim_p_pe * head_dim_p_pe), dtype=np.float16)
        for i in range(P):
            for j in range(P):
                t = ind[i, j]
                for h in range(n_h):
                    blk = Wmat[t * dim_p_pe:(t + 1) * dim_p_pe,
                              h * head_dim + j * head_dim_p_pe: h * head_dim + (j + 1) * head_dim_p_pe]
                    off = h * dim_p_pe * head_dim_p_pe
                    out[i, j, off:off + dim_p_pe * head_dim_p_pe] = blk.reshape(-1)
        return out

    def shard_o_headmajor(Wo_, n_h):   # Wo_: [n_h*head_dim, dim]
        out = np.zeros((P, P, n_h * head_dim_p_pe * dim_p_pe), dtype=np.float16)
        for i in range(P):
            for j in range(P):
                t = ind[i, j]
                for h in range(n_h):
                    blk = Wo_[h * head_dim + t * head_dim_p_pe: h * head_dim + (t + 1) * head_dim_p_pe,
                             j * dim_p_pe:(j + 1) * dim_p_pe]
                    off = h * head_dim_p_pe * dim_p_pe
                    out[i, j, off:off + head_dim_p_pe * dim_p_pe] = blk.reshape(-1)
        return out

    q_hm = shard_qkv_headmajor(Wq, n_heads)
    k_hm = shard_qkv_headmajor(Wk, n_kv_heads)
    v_hm = shard_qkv_headmajor(Wv, n_kv_heads)
    o_hm = shard_o_headmajor(Wo, n_heads)

    # ----- FFN weights: simple shifted sharding -----
    up_sh = np.zeros((dim, ffn_dim), dtype=np.float16)
    gate_sh = np.zeros((dim, ffn_dim), dtype=np.float16)
    down_sh = np.zeros((ffn_dim, dim), dtype=np.float16)
    for i in range(P):
        for j in range(P):
            t = ind[i, j]
            up_sh[i * dim_p_pe:(i + 1) * dim_p_pe, j * ffn_dim_p_pe:(j + 1) * ffn_dim_p_pe] = \
                Wup[t * dim_p_pe:(t + 1) * dim_p_pe, j * ffn_dim_p_pe:(j + 1) * ffn_dim_p_pe]
            gate_sh[i * dim_p_pe:(i + 1) * dim_p_pe, j * ffn_dim_p_pe:(j + 1) * ffn_dim_p_pe] = \
                Wgate[t * dim_p_pe:(t + 1) * dim_p_pe, j * ffn_dim_p_pe:(j + 1) * ffn_dim_p_pe]
            down_sh[i * ffn_dim_p_pe:(i + 1) * ffn_dim_p_pe, j * dim_p_pe:(j + 1) * dim_p_pe] = \
                Wdown[t * ffn_dim_p_pe:(t + 1) * ffn_dim_p_pe, j * dim_p_pe:(j + 1) * dim_p_pe]

    # RoPE OFF: cos=1, sin=0 (kernel rope becomes a pair-swap, cancels in the score)
    tensor_freqs_cos = np.ones((P, P, _dim_p_pe // 2), dtype=np.float16)
    tensor_freqs_sin = np.zeros((P, P, _dim_p_pe // 2), dtype=np.float16)

    # ============================ run on sim ============================
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = os.path.abspath(f"out_{cfg_name}")
    if not os.path.isdir(out_dir):
        raise SystemExit(f"Host: {out_dir} not found — run compile.py --mode sim first")
    os.chdir(out_dir)
    runner = SdkRuntime(out_dir)
    runner.load()
    runner.run()

    # Pre-fetch all symbol handles right after run() (the launch_sim.py ordering).
    names = ["X", "W", "W2", "Q_weight", "K_weight", "V_weight", "O_weight",
             "freqs_sin", "freqs_cos", "UP_weight", "GATE_weight", "DOWN_weight",
             "Z", "Z_mid", "score_dbg", "xk_trace", "probs_dbg", "xv_dbg",
             "output_dbg", "h1_dbg"]
    syms = {}
    for nm in names:
        syms[nm] = runner.get_id(nm)
        if syms[nm] is None:
            print(f"WARNING: symbol '{nm}' not found in artifact (get_id -> None)")

    def h2d(sym_name, arr, elems):
        u32 = input_array_to_u32(np.ascontiguousarray(arr).ravel(), 1, 1)
        runner.memcpy_h2d(syms[sym_name], u32, 0, 0, P, P, elems, streaming=False,
                          data_type=io_dtype, order=memcpy_order, nonblock=False)

    Xc = X.reshape(P, seq_len_p_pe, P, dim_p_pe).transpose(0, 2, 3, 1).reshape(P, P, seq_len_p_pe * dim_p_pe)
    h2d("X", Xc, seq_len_p_pe * dim_p_pe)
    h2d("W", tensor_W, dim_p_pe)
    h2d("W2", tensor_W2, dim_p_pe)
    h2d("Q_weight", q_hm, n_heads * dim_p_pe * head_dim_p_pe)
    h2d("K_weight", k_hm, n_kv_heads * dim_p_pe * head_dim_p_pe)
    h2d("V_weight", v_hm, n_kv_heads * dim_p_pe * head_dim_p_pe)
    h2d("O_weight", o_hm, n_heads * head_dim_p_pe * dim_p_pe)
    h2d("freqs_sin", tensor_freqs_sin, _dim_p_pe // 2)
    h2d("freqs_cos", tensor_freqs_cos, _dim_p_pe // 2)

    UP = up_sh.reshape(P, dim_p_pe, P, ffn_dim_p_pe).transpose(0, 2, 1, 3).reshape(P, P, dim_p_pe * ffn_dim_p_pe)
    h2d("UP_weight", UP, dim_p_pe * ffn_dim_p_pe)
    GATE = gate_sh.reshape(P, dim_p_pe, P, ffn_dim_p_pe).transpose(0, 2, 1, 3).reshape(P, P, dim_p_pe * ffn_dim_p_pe)
    h2d("GATE_weight", GATE, dim_p_pe * ffn_dim_p_pe)
    DOWN = down_sh.reshape(P, ffn_dim_p_pe, P, dim_p_pe).transpose(0, 2, 1, 3).reshape(P, P, ffn_dim_p_pe * dim_p_pe)
    h2d("DOWN_weight", DOWN, ffn_dim_p_pe * dim_p_pe)

    runner.launch('init_task', nonblock=False)
    runner.launch('prefill_host', np.int16(1), np.int16(1), nonblock=False)

    def d2h(sym_name, elems):
        if syms.get(sym_name) is None:
            return None
        buf = np.zeros(P * P * elems, dtype=np.uint32)
        runner.memcpy_d2h(buf, syms[sym_name], 0, 0, P, P, elems, streaming=False,
                          order=memcpy_order, data_type=io_dtype, nonblock=False)
        return memcpy_view(buf, np.dtype(np.float16))

    try:
        Zmid_1d = d2h("Z_mid", seq_len_p_pe * dim_p_pe)
        Zmid = untile_flat_1d(Zmid_1d, P, seq_len_p_pe, dim_p_pe)     # [seq, dim]
        score_dbg_1d = d2h("score_dbg", seq_len_p_pe * seq_len_p_pe)
        xk_trace_1d = d2h("xk_trace", P * seq_len_p_pe * head_dim_p_pe)
        probs_dbg_1d = d2h("probs_dbg", seq_len_p_pe * seq_len_p_pe)
        xv_dbg_1d = d2h("xv_dbg", seq_len_p_pe * head_dim_p_pe)
        output_dbg_1d = d2h("output_dbg", seq_len_p_pe * head_dim_p_pe)
        h1_dbg_1d = d2h("h1_dbg", seq_len_p_pe * dim_p_pe)
        Zfull_1d = d2h("Z", seq_len_p_pe * dim_p_pe)
        Zfull = untile_flat_1d(Zfull_1d, P, seq_len_p_pe, dim_p_pe) if Zfull_1d is not None else None
    finally:
        runner.stop()

    # ============================ numpy oracle ============================
    Xf = X.astype(np.float32)
    ms = np.mean(Xf ** 2, axis=1, keepdims=True)          # sum(x^2)/dim
    Xn = Xf / np.sqrt(ms + eps)                           # W = ones
    attn = np.zeros((seq_len, dim), dtype=np.float32)
    causal = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)  # True where key>query
    raw_score_h0 = None
    for h in range(n_heads):
        g = h // group
        Qh = Xn @ Wq[:, h * head_dim:(h + 1) * head_dim].astype(np.float32)
        Kg = Xn @ Wk[:, g * head_dim:(g + 1) * head_dim].astype(np.float32)
        Vg = Xn @ Wv[:, g * head_dim:(g + 1) * head_dim].astype(np.float32)
        raw = Qh @ Kg.T
        if h == 0:
            raw_score_h0 = raw.copy()
            Vg_h0 = Vg.copy()
        S = raw * alpha
        S[causal] = -np.inf
        S = S - S.max(axis=1, keepdims=True)
        e = np.exp(S)
        Pr = e / e.sum(axis=1, keepdims=True)
        if h == 0:
            probs_h0 = Pr.copy()
        Oh = Pr @ Vg
        if h == 0:
            Oh_h0 = Oh.copy()
            h1_h0 = Oh @ Wo[h * head_dim:(h + 1) * head_dim, :].astype(np.float32)
        attn += Oh @ Wo[h * head_dim:(h + 1) * head_dim, :].astype(np.float32)
    Zmid_oracle = Xf + attn

    # ============================ compare ============================
    print("\n--- Z_mid (post-attention residual = X + attn) ---")
    report("Z_mid (full, incl. X passthrough)", Zmid, Zmid_oracle)
    # The honest attention metric: strip the X passthrough and compare only attn.
    print("\n--- attention CONTRIBUTION  (Z_mid - X) ---")
    report("attn-contrib", Zmid.astype(np.float64) - Xf, Zmid_oracle - Xf)
    print(f"   |attn_kernel|={np.linalg.norm(Zmid.astype(np.float64)-Xf):.4f}  "
          f"|attn_oracle|={np.linalg.norm(Zmid_oracle-Xf):.4f}")

    # ---- verbose array dumps (P is small) ----
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    attn_k = Zmid.astype(np.float64) - Xf
    attn_o = Zmid_oracle - Xf
    print("\nkernel attn (Z_mid - X), rows 0..3:")
    print(attn_k[:4])
    print("row L2 norms (kernel):", np.round(np.linalg.norm(attn_k, axis=1), 4))
    print("oracle attn (Z_mid_oracle - X), rows 0..3:")
    print(attn_o[:4])
    print("row L2 norms (oracle):", np.round(np.linalg.norm(attn_o, axis=1), 4))
    # did the kernel run the pipeline at all? |Z_full| (after FFN) and |Z_mid|
    if Zfull is not None:
        print(f"\n|Z_full (after FFN)|={np.linalg.norm(Zfull):.4f}  "
              f"|Z_mid|={np.linalg.norm(Zmid):.4f}  |X|={np.linalg.norm(Xf):.4f}")

    # ========================= FFN =========================
    # Two lessons carried over from the attention debug:
    #  (1) The aggregate cosine LIES. |Z_full| ~ |Z_mid| + a small correction, so the full-Z
    #      cosine hides FFN error exactly as Z_mid 0.95 hid attention at 0.16. Use the
    #      FFN-CONTRIBUTION cosine, cos(Z - Z_mid, ffn_ref(Z_mid)).
    #  (2) Drive the oracle from the KERNEL's own Z_mid, so attention error cannot leak in
    #      and the FFN is tested in isolation.
    # And the specific hypothesis: z2_matmul (gate) is structurally identical to the
    # xk_matmul buffer-parity bug -- it hardcodes Z_norm_tile and skips the pre-shift,
    # trusting z1_matmul to have left the shifted Z there. If that mechanism applies, the
    # error must split by offset_step PARITY, per row.
    if Zfull is not None:
        def ffn_ref(z):
            z = z.astype(np.float64)
            ms_ = np.mean(z ** 2, axis=1, keepdims=True)
            zn = z / np.sqrt(ms_ + eps)                     # rmsnorm_z (W2 = ones)
            up = zn @ Wup.astype(np.float64)                # z1_matmul
            gate = zn @ Wgate.astype(np.float64)            # z2_matmul
            z3 = (gate / (1.0 + np.exp(-gate))) * up        # z3_comp: silu(gate) * up
            return z3 @ Wdown.astype(np.float64)            # h2_matmul

        Zmid_k = Zmid.astype(np.float64)
        ffn_oracle = ffn_ref(Zmid_k)
        ffn_kernel = Zfull.astype(np.float64) - Zmid_k

        print("\n=========================== FFN ===========================")
        print("--- full Z (incl. Z_mid passthrough -- CAN HIDE FFN ERROR) ---")
        report("Z_full", Zfull.astype(np.float64), Zmid_k + ffn_oracle)
        print("\n--- FFN CONTRIBUTION (Z - Z_mid); oracle driven by the KERNEL's Z_mid ---")
        report("ffn-contrib", ffn_kernel, ffn_oracle)
        print(f"   |ffn_kernel|={np.linalg.norm(ffn_kernel):.4f}  "
              f"|ffn_oracle|={np.linalg.norm(ffn_oracle):.4f}")

        # per-row breakdown against offset_step parity -- the z2_matmul hypothesis
        def _os(py):
            if py == 0:
                return 0
            if py % 2 == 0:
                return P - (py // 2)
            return (py + 1) // 2

        row_err = np.abs(ffn_kernel - ffn_oracle).max(axis=1)
        row_cos = np.array([
            float(np.dot(ffn_kernel[r], ffn_oracle[r]) /
                  (np.linalg.norm(ffn_kernel[r]) * np.linalg.norm(ffn_oracle[r]) + 1e-30))
            for r in range(seq_len)])
        os_of_row = [_os(r // seq_len_p_pe) for r in range(seq_len)]

        print("\n  per-row FFN diagnosis (z2_matmul buffer-parity hypothesis):")
        print("   row | py | offset_step | parity | max_abs_err | row cosine")
        for r in range(seq_len):
            o = os_of_row[r]
            print(f"   {r:3d} | {r // seq_len_p_pe:2d} |     {o:3d}     |  "
                  f"{'even' if o % 2 == 0 else 'odd '}  |  {row_err[r]:9.4f}  |  {row_cos[r]:.6f}")

        ev = [row_cos[r] for r in range(seq_len) if os_of_row[r] % 2 == 0]
        od = [row_cos[r] for r in range(seq_len) if os_of_row[r] % 2 == 1]
        if ev and od:
            print(f"\n  mean row cosine -- even offset_step: {np.mean(ev):.6f}   "
                  f"odd: {np.mean(od):.6f}")
            if np.mean(od) < 0.9 <= np.mean(ev):
                print("  => CONFIRMED: z2_matmul has the SAME buffer-parity bug xk_matmul had.")
                print("     Fix identically: swap into recv rather than naming Z_norm_tile.")
            elif min(np.mean(ev), np.mean(od)) > 0.99:
                print("  => FFN is clean per-row: NO buffer-parity bug. Any remaining gap is")
                print("     fp16 accumulation, and the 0.986 device figure is honest.")
            else:
                print("  => FFN error does NOT split by offset_step parity -- a different bug.")

    # ---- output_matmul operands: probs (softmax+causal mask) and XV (V projection) ----
    # The raw score is validated, so if attention is still wrong the fault is in one of
    # these two, or in output_matmul/O-proj if both are clean.
    if seq_len_p_pe == 1 and head_dim_p_pe == 1:
        # rope runs with cos=1/sin=0 (identity), but tolerate an adjacent-pair column swap
        sw_v = np.arange(head_dim)
        sw_v[0::2], sw_v[1::2] = np.arange(1, head_dim, 2), np.arange(0, head_dim, 2)

        if probs_dbg_1d is not None:
            probs_k = probs_dbg_1d.reshape(P, P).astype(np.float64)     # [query=py, key=px]
            print("\n--- head-0 probs (post-softmax, post-causal-mask) ---")
            report("probs", probs_k, probs_h0)
            print("kernel probs [py, px]:")
            print(probs_k)
            print("oracle probs [query, key]:")
            print(probs_h0.astype(np.float64))
            # The causal mask must zero every key > query. Check that structurally: a
            # nonzero upper triangle means the mask is masking the wrong keys (or not at
            # all); rows not summing to 1 mean the softmax normalisation is off.
            upper = np.triu(np.ones((P, P), dtype=bool), k=1)
            print(f"  max prob above the diagonal (must be ~0): {np.abs(probs_k[upper]).max():.4g}")
            print("  row sums (must all be 1.0):", np.round(probs_k.sum(axis=1), 4))

        if xv_dbg_1d is not None:
            xv_k = xv_dbg_1d.reshape(P, P).astype(np.float64)           # [py, px]
            print("\n--- head-0 XV (V projection, as output_matmul consumes it) ---")
            report("XV", xv_k, Vg_h0.astype(np.float64))
            e_direct = np.abs(xv_k - Vg_h0).max(axis=1)
            e_swap = np.abs(xv_k - Vg_h0[:, sw_v]).max(axis=1)
            print("  per-row max|XV - V|:", np.round(np.minimum(e_direct, e_swap), 4))
            print("  offset_step(py)    :", [0 if py == 0 else (P - py // 2 if py % 2 == 0
                                             else (py + 1) // 2) for py in range(P)])

    # ---- the last three stages: output_matmul, h1_matmul (O-proj), z_add ----
    # probs and XV are validated, so if output_h is wrong the fault is output_matmul;
    # if output_h is right but h1 is wrong it is the O-proj; if both are right the fault
    # is z_add or the cross-head accumulation into h1.
    if seq_len_p_pe == 1 and head_dim_p_pe == 1 and output_dbg_1d is not None:
        out_k = output_dbg_1d.reshape(P, P).astype(np.float64)       # [py=query, px=dim]
        print("\n--- head-0 output_h = probs @ XV (input to the O-proj) ---")
        report("output_h", out_k, Oh_h0.astype(np.float64))
        e_direct = np.abs(out_k - Oh_h0).max(axis=1)
        e_swap = np.abs(out_k - Oh_h0[:, sw_v]).max(axis=1)
        print("  per-row max|output_h - Oh|:", np.round(np.minimum(e_direct, e_swap), 4))

        # ---- BUG 2 hypothesis, tested exactly ----
        # A mesh GEMM needs right[i,j] to hold contraction block ind[i,j]; the host does this
        # for every weight (t = ind[i,j] in the shard fns). XV is computed on-chip and stays
        # natural, so the GEMM effectively multiplies by B_eff[m,j] = V[indinv[m,j], j].
        # If the measured output matches probs @ B_eff, the mechanism is proven and the fix is
        # exactly: make XV[i,j] hold V[ind[i,j], j] before output_matmul.
        indinv = np.zeros((P, P), dtype=int)
        for j in range(P):
            for i in range(P):
                indinv[ind[i, j], j] = i
        B_eff = np.zeros((P, P), dtype=np.float64)
        for m in range(P):
            for j in range(P):
                B_eff[m, j] = Vg_h0[indinv[m, j], j]
        pred = probs_h0.astype(np.float64) @ B_eff
        print("\n--- BUG 2 hypothesis: kernel computes probs @ V[indinv] (right operand unskewed) ---")
        report("predicted output_h", out_k, pred)
        print("  PASS here => confirmed: output_matmul needs XV[i,j] = V[ind[i,j], j].")
        print("  FAIL here => the ind model is wrong; do NOT build the skew yet.")

    if seq_len_p_pe == 1 and h1_dbg_1d is not None:
        h1_k = untile_flat_1d(h1_dbg_1d, P, seq_len_p_pe, dim_p_pe)  # [seq, dim]
        print("\n--- h1 after head-0 O-proj only (head 0's contribution to attn) ---")
        report("h1(head0)", h1_k.astype(np.float64), h1_h0.astype(np.float64))
        print("  per-row max|h1 - Oh@Wo_h0|:",
              np.round(np.abs(h1_k.astype(np.float64) - h1_h0).max(axis=1), 4))
        print(f"  |h1_kernel|={np.linalg.norm(h1_k):.4f}  |h1_oracle|={np.linalg.norm(h1_h0):.4f}")

        # Cross-check the ind model on a GEMM that WORKS: h1 = output_h @ Wo, whose right
        # operand IS host-skewed. Predict h1 from the kernel's own (wrong) output_h -- if it
        # matches, the O-proj is correct and merely inherits output_matmul's error, and the
        # "host-skewed right operand => GEMM is correct" model holds.
        if head_dim_p_pe == 1 and output_dbg_1d is not None:
            h1_pred = out_k @ Wo[0:head_dim, :].astype(np.float64)
            print("\n--- cross-check: h1 should equal (kernel's own output_h) @ Wo_h0 ---")
            report("h1 from kernel output_h", h1_k.astype(np.float64), h1_pred)
            print("  PASS => O-proj correct, only inherits output_matmul's error.")

    # ---- XK delivery trace: does every column hold the SAME key at each step? ----
    # The reduce sums partials across px, which is only meaningful if all columns in a
    # row hold chunks of one key row. Swapping the reduce for the validated fixed-root
    # all-reduce left the even/odd fingerprint identical => the partials are suspect.
    # Each trace value is matched back to its (key, head_dim col) by nearest value, so
    # this is robust to any column permutation the rope/sharding may introduce.
    if seq_len_p_pe == 1 and head_dim_p_pe == 1 and xk_trace_1d is not None:
        xk = xk_trace_1d.reshape(P, P, P).astype(np.float64)          # [py, px, step]
        Kg0 = (Xn @ Wk[:, 0:head_dim].astype(np.float32)).astype(np.float64)   # [key, head_dim]
        flat = Kg0.ravel()                                            # idx = key*head_dim + d

        def which_key(v):
            j = int(np.argmin(np.abs(flat - v)))
            return j // head_dim, float(abs(flat[j] - v))

        def offset_step_of(py):
            if py == 0:
                return 0
            if py % 2 == 0:
                return P - (py // 2)
            return (py + 1) // 2

        f = reduce_root_map(P)
        seen = np.full((P, P), -1, dtype=int)     # [py, step] key, or -1 if columns disagree
        want = np.zeros((P, P), dtype=int)        # [py, step] key the deposit column implies
        worst = 0.0
        for py in range(P):
            for s in range(P):
                ks = []
                for px in range(P):
                    k, e = which_key(xk[py, px, s])
                    ks.append(k)
                    worst = max(worst, e)
                seen[py, s] = ks[0] if len(set(ks)) == 1 else -1
                want[py, s] = int(f[(offset_step_of(py) + s) % P])

        print("\n--- XK delivery trace (head 0) ---")
        print("key each row holds at each step (-1 = columns DISAGREE):  [py, step]")
        print(seen)
        print("key the deposit column f(co) implies it should be:        [py, step]")
        print(want)
        print(f"(worst nearest-value match residual: {worst:.4g} -- large means the")
        print(" identification itself is unreliable, not the alignment)")

        # At step 0 no shift has happened yet: every row must simply hold its own K row.
        # If rows are already wrong here, the K PROJECTION is broken and the score shift
        # is innocent (it just carries corrupt rows around). rope runs with cos=1/sin=0,
        # which is identity, but check an adjacent-pair column swap too just in case.
        sw = np.arange(head_dim)
        sw[0::2], sw[1::2] = np.arange(1, head_dim, 2), np.arange(0, head_dim, 2)
        xk0 = xk[:, :, 0]                                  # [py, px] at step 0
        e_direct = np.abs(xk0 - Kg0).max(axis=1)           # per row
        e_swap = np.abs(xk0 - Kg0[:, sw]).max(axis=1)
        per_row = np.minimum(e_direct, e_swap)
        print("\n--- K projection sanity: XK at step 0 must equal K[py, :] (no shift yet) ---")
        print("  per-row max|XK(step0) - K|:", np.round(per_row, 4))
        print("  offset_step(py)          :", [offset_step_of(py) for py in range(P)])
        bad_rows = [py for py in range(P) if per_row[py] > 5e-2]
        print(f"  rows with a BROKEN K projection: {bad_rows}")
        if bad_rows:
            odd_off = [py for py in range(P) if offset_step_of(py) % 2 == 1]
            print(f"  rows with ODD offset_step (odd X pre-shift hop count): {odd_off}")
            if sorted(bad_rows) == sorted(odd_off):
                print("  => CONFIRMED: the K projection is wrong exactly for rows needing an ODD")
                print("     pre-shift hop count. The bug is in xk_matmul / left_matrix_shift,")
                print("     NOT in the score matmul or its reduce.")
        else:
            print("  => K projection is CLEAN; the corruption really does start in the shift.")

        # Is an odd (mixed) step per-column DRIFT (each column holds some *valid* K row,
        # so the PEs are out of lockstep) or GARBAGE (values match no K row, so we are
        # reading a buffer mid-write)? The residual separates the two.
        print("\nper-column detail for py=0 (key@residual; residual >> 0 => not a real K row):")
        for s in range(P):
            co = (offset_step_of(0) + s) % P
            cells = []
            for px in range(P):
                k, e = which_key(xk[0, px, s])
                cells.append(f"{k}@{e:.3f}")
            print(f"  step {s} (co={co},{'even' if co % 2 == 0 else 'odd '}): " + "  ".join(cells))

        odd_res = [which_key(xk[py, px, s])[1]
                   for py in range(P) for s in range(P) for px in range(P)
                   if (offset_step_of(py) + s) % P % 2 == 1]
        even_res = [which_key(xk[py, px, s])[1]
                    for py in range(P) for s in range(P) for px in range(P)
                    if (offset_step_of(py) + s) % P % 2 == 0]
        print(f"\n  residual on even-co steps: max={max(even_res):.4g} mean={np.mean(even_res):.4g}")
        print(f"  residual on odd-co  steps: max={max(odd_res):.4g} mean={np.mean(odd_res):.4g}")
        # NB: a mean residual ~ (K value range)/(number of K entries) is what RANDOM values
        # score against this table, so a nonzero odd-step residual only says "not a real K
        # row" -- it does NOT distinguish drift from tearing. Trust the step-0 check above.
        if max(odd_res) < 1e-2:
            print("  => odd-step values ARE real K rows, just different ones per column.")
        else:
            print("  => odd-step values match no real K row (consistent with random data).")

        mixed = int((seen < 0).sum())
        misaligned = int(((seen >= 0) & (seen != want)).sum())
        print(f"\n  steps where columns disagree on the key: {mixed}/{P*P}")
        print(f"  steps aligned but on the WRONG key:       {misaligned}/{P*P}")
        if mixed:
            bad_odd = all(seen[py, s] < 0
                          for py in range(P) for s in range(P)
                          if (offset_step_of(py) + s) % P % 2 == 1)
            good_even = all(seen[py, s] >= 0
                            for py in range(P) for s in range(P)
                            if (offset_step_of(py) + s) % P % 2 == 0)
            print(f"  all odd-co steps mixed: {bad_odd};  all even-co steps clean: {good_even}")
            if bad_odd and good_even:
                print("  => a row is clean exactly when co is even, i.e. exactly when it currently")
                print("     holds one of the keys whose projection survived -- see the step-0 check")
                print("     above for whether the shift or the projection is at fault.")
        elif misaligned:
            print("  => XK delivery is CLEAN; the key->column mapping f(co) is what's wrong.")
        else:
            print("  => XK delivery is clean AND correctly aligned; look elsewhere.")

    if seq_len_p_pe == 1 and score_dbg_1d is not None:
        # score_dbg: PE(py,px) holds one raw score; grid[py,px] = score[query=py, key=?]
        score_grid = score_dbg_1d.reshape(P, P).astype(np.float64)   # [py, px]
        print("\nkernel score_grid [py, px]:")
        print(score_grid)
        print("oracle raw score [query, key] (head 0):")
        print(raw_score_h0.astype(np.float64))
        oracle_raw = raw_score_h0.astype(np.float64)                 # [query, key]
        print("\n--- head-0 raw score (Q.K^T, pre-alpha/mask) ---")
        report("score (assume col px == key)", score_grid, oracle_raw)
        # is each kernel row a permutation of the oracle row?
        perm_ok = True
        for q in range(P):
            if not np.allclose(np.sort(score_grid[q]), np.sort(oracle_raw[q]), atol=2e-2):
                perm_ok = False
                break
        print(f"score rows are a permutation of the oracle rows: {perm_ok}")

        # ---- per-column diagnosis: correlate error with the reduce_root/co of each column ----
        f = reduce_root_map(P)                 # f[co] = column
        finv = np.zeros(P, dtype=int)
        finv[f] = np.arange(P)                 # finv[col] = co that deposits into col
        col_err = np.abs(score_grid - oracle_raw).mean(axis=0)   # mean |err| per column over queries
        print("\nper-column diagnosis (does the deposit column hold the right key's score?):")
        print("  col | co=finv(col) | co_parity | mean_abs_err | ok?")
        for px in range(P):
            co = int(finv[px])
            ok = "OK " if col_err[px] < 0.05 else "BAD"
            print(f"   {px:2d} |     {co:2d}       |   {'even' if co % 2 == 0 else 'odd '} "
                  f"    |   {col_err[px]:7.4f}   | {ok}")
        good_even = all(col_err[f[co]] < 0.05 for co in range(P) if co % 2 == 0)
        bad_odd = all(col_err[f[co]] >= 0.05 for co in range(P) if co % 2 == 1)
        print(f"\n  => even-co columns all correct: {good_even};  odd-co columns all wrong: {bad_odd}")
        if good_even and bad_odd:
            print("  CONFIRMED: matmul_T reduce deposits correctly only on EVEN co; the transposed-")
            print("  score reduce (matmul_T_reduce_add_x) is wrong for odd-co reduce_roots at")
            print("  seq_len_p_pe==1. Correct roots:", sorted(int(f[co]) for co in range(P) if co % 2 == 0))
        if perm_ok:
            print("  -> per-row values match under a COLUMN PERMUTATION.")
            print("     If the direct comparison FAILED but this is True, the score math is")
            print("     correct but columns are permuted -> the causal mask (which keys off")
            print("     physical px) is masking the WRONG keys. Recovered col->key per row:")
            for q in range(P):
                perm = []
                for px in range(P):
                    diffs = np.abs(oracle_raw[q] - score_grid[q, px])
                    perm.append(int(np.argmin(diffs)))
                print(f"    query py={q}: column->key = {perm}")
    else:
        print("\n(seq_len_p_pe != 1: skipping the per-column score-order check)")


if __name__ == "__main__":
    main()
