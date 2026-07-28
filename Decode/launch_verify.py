"""launch_verify.py — validate the Decode kernel against transformer_lens (real Llama-3-8B block 0).

Unlike launch_sim.py (synthetic weights + numpy oracle at tiny dims), this loads REAL
Meta-Llama-3-8B block-0 weights, seeds the KV cache from the transformer_lens oracle
(pytorch/oracle_decode/), runs ONE decode step at position `prefill_len`, and compares the
kernel's resid_mid / resid_post to the oracle via cosine similarity.

Oracle key insight (baseline_decode.py): causal attention ⇒ a full-sequence forward pass at
position t equals an autoregressive decode step at t. So decode at t=prefill_len takes
input resid_pre[t], a cache seeded with K/V[:t], and should reproduce resid_mid[t]/resid_post[t].

Run (after compiling the kernel with the same config, e.g. via run_verify.sh):
    cs_python launch_verify.py --config model_config/llama8B_block0_p32.json
"""
import os
import json
import glob
import argparse
import numpy as np

# Appliance/cluster device flow (mirrors Prefill/launch_device.py): cloud-compiled artifact +
# cerebras.sdk.client.SdkRuntime(simulator=...). EIDF eidf002-cs3 is appliance-mode (no cmaddr).
from cerebras.sdk.client import SdkRuntime, sdk_utils
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder

THETA = 500000.0  # Llama-3 RoPE base
OUT_PATH = "compile_out"  # cloud artifacts written by `python compile.py --mode device`


# Decode-layout NUMPY helpers, inlined from launch_sim.py so this device harness NEVER imports the
# pybind/simfab module `cerebras.sdk.sdk_utils` (absent in the appliance venv). Mirrors Prefill's
# standalone launch_device.py.
def sep(title):
    print(f"\n{'═'*64}")
    print(f"  {title}")
    print(f"{'═'*64}")


def tile_kcache_interleaved(K_cache, P, bsz, kv_dim_p_pe, max_seq_len_p_pe, prefill_len_p_pe):
    """K_cache[bsz, kv_dim, max_seq_len] -> KCache_tile[P, P, bsz*kv_dim_p_pe*max_seq_len_p_pe]
    (round-robin interleaved: token t -> py=t%P, slot=t//P)."""
    cache_per_pe = kv_dim_p_pe * max_seq_len_p_pe
    tile = np.zeros((P, P, bsz * cache_per_pe), dtype=np.float16)
    for b in range(bsz):
        for py in range(P):
            for px in range(P):
                for s in range(prefill_len_p_pe):
                    token = py + s * P
                    k_row_start = px * kv_dim_p_pe
                    for k in range(kv_dim_p_pe):
                        tile[py, px, b * cache_per_pe + k * max_seq_len_p_pe + s] = \
                            K_cache[b, k_row_start + k, token]
    return tile


def tile_vcache_interleaved(V_cache, P, bsz, kv_dim_p_pe, max_seq_len_p_pe, prefill_len_p_pe):
    """V_cache[bsz, max_seq_len, kv_dim] -> VCache_tile[P, P, bsz*max_seq_len_p_pe*kv_dim_p_pe]."""
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


def cast_tensor_u32(t):
    """fp16 tensor -> u32 view (memcpy transfers use 32-bit buffers); matches Prefill launch_device."""
    return np.uint32(t.view(np.uint16))


def d2h(runner, sym_id, P, bsz, data_per_pe, io_dtype, memcpy_order):
    """Client-API read of P×P PEs, each contributing bsz*data_per_pe f16 -> grid [P, P, count]
    (same shape/order as launch_sim.d2h so reconstruct() is unchanged)."""
    count = bsz * data_per_pe
    buf = np.zeros(P * P * count, dtype=np.uint32)
    runner.memcpy_d2h(buf, sym_id, 0, 0, P, P, count,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)
    return sdk_utils.memcpy_view(buf, np.dtype(np.float16)).reshape(P, P, count)


# ─────────────────────────────────────────────────────────────────────────────
# Real Meta-Llama-3-8B block-0 weights (mirrors Prefill/launch_device.py)
# ─────────────────────────────────────────────────────────────────────────────
def rope_perm_idx(D):
    """Within-head column permutation mapping kernel adjacent-pair rope to HF rotate-half:
    kernel col 2i <- HF col i+D/2, kernel col 2i+1 <- HF col i."""
    half = D // 2
    p = np.empty(D, dtype=np.int64)
    p[0::2] = np.arange(half) + half
    p[1::2] = np.arange(half)
    return p


def rope_hf(x, pos, theta=THETA):
    """HF rotate-half RoPE on x[..., head_dim] at scalar position `pos` (same as verify_numpy)."""
    hd = x.shape[-1]; half = hd // 2
    inv = theta ** (-2.0 * np.arange(half) / hd)
    ang = pos * inv
    cos_f = np.concatenate([np.cos(ang), np.cos(ang)])
    sin_f = np.concatenate([np.sin(ang), np.sin(ang)])
    x1, x2 = x[..., :half], x[..., half:]
    rot = np.concatenate([-x2, x1], axis=-1)
    return x * cos_f + rot * sin_f


def load_block0_weights(head_dim, n_heads, n_kv_heads):
    """HF Linear stores [out, in] and applies x@W.T; WaferLLM does x@W, so 2-D weights are
    transposed. q/k get rope_perm (HF rotate-half -> kernel adjacent pairs); 1-D norms as-is."""
    import torch
    from safetensors import safe_open

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_dir = os.path.join(hf_home, "hub", "models--meta-llama--Meta-Llama-3-8B")
    prefix = "model.layers.0."

    raw = {}
    for shard in sorted(glob.glob(os.path.join(model_dir, "**", "*.safetensors"), recursive=True)):
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                if name.startswith(prefix):
                    raw[name] = f.get_tensor(name).to(torch.float32).cpu().numpy().astype(np.float16)

    def get(suffix, transpose=True):
        w = raw[prefix + suffix]
        return w.T if transpose else w

    p = rope_perm_idx(head_dim)

    def rope_perm(w, n):
        full = np.concatenate([h * head_dim + p for h in range(n)])
        return w[:, full]

    return {
        "q":         rope_perm(get("self_attn.q_proj.weight"), n_heads),
        "k":         rope_perm(get("self_attn.k_proj.weight"), n_kv_heads),
        "v":         get("self_attn.v_proj.weight"),
        "o":         get("self_attn.o_proj.weight"),
        "up":        get("mlp.up_proj.weight"),
        "gate":      get("mlp.gate_proj.weight"),
        "down":      get("mlp.down_proj.weight"),
        "norm_pre":  get("input_layernorm.weight", transpose=False),
        "norm_post": get("post_attention_layernorm.weight", transpose=False),
    }


def load_raw_attn_weights():
    """Raw HF block-0 attention weights (NO rope_perm, NO transpose) as fp32 — the head-space
    ground truth for --dump-intermediates, exactly as verify_numpy loads them. HF stores [out,in]
    and applies y = x @ W.T; norm is 1-D."""
    import torch
    from safetensors import safe_open
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_dir = os.path.join(hf_home, "hub", "models--meta-llama--Meta-Llama-3-8B")
    prefix = "model.layers.0."
    want = {"q": "self_attn.q_proj.weight", "k": "self_attn.k_proj.weight",
            "v": "self_attn.v_proj.weight", "norm": "input_layernorm.weight"}
    raw = {}
    for shard in sorted(glob.glob(os.path.join(model_dir, "**", "*.safetensors"), recursive=True)):
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                if name.startswith(prefix):
                    raw[name] = f.get_tensor(name).to(torch.float32).cpu().numpy()
    return {k: raw[prefix + s] for k, s in want.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Reporting (cosine; contribution cosine strips the residual passthrough)
# ─────────────────────────────────────────────────────────────────────────────
def _cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def report(name, got, ref):
    cos = _cos(got, ref)
    abs_err = np.abs(got.astype(np.float32) - ref.astype(np.float32))
    print(f"  [{'PASS' if cos >= 0.999 else 'FAIL'}] {name:28s} full cosine={cos:.6f}  "
          f"max_abs={abs_err.max():.3e}  mean_abs={abs_err.mean():.3e}")
    return cos >= 0.999


def report_contrib(name, got, ref, base):
    """Honest metric: a small block correction rides a large residual passthrough, so the full
    cosine flatters a wrong block. Strip the passthrough (base) and compare only the contribution."""
    g = got.astype(np.float64) - base.astype(np.float64)
    r = ref.astype(np.float64) - base.astype(np.float64)
    cos = _cos(g, r)
    print(f"  [{'PASS' if cos >= 0.999 else 'FAIL'}] {name:28s} contrib cosine={cos:.6f}  "
          f"|kernel|={np.linalg.norm(g):.4f}  |ref|={np.linalg.norm(r):.4f}")
    return cos >= 0.999


def parse_args():
    ap = argparse.ArgumentParser(description="Validate Decode kernel vs transformer_lens (real Llama-3-8B block 0)")
    ap.add_argument("--config", default="model_config/llama8B_block0_p32.json")
    ap.add_argument("--oracle-dir", default=None, help="dir with resid_*_block0.npy + k/v_cache (default: <repo>/pytorch/oracle_decode)")
    ap.add_argument("--simulator", action="store_true",
                    help="run the cloud artifact in the appliance SIMULATOR (default: real WSE-3)")
    ap.add_argument("--save-zmid", default="csl_decode_zmid.npy",
                    help="path to write the DEVICE resid_mid (Z_mid), threaded into the FFN stage")
    ap.add_argument("--dump-intermediates", action="store_true",
                    help="read back the kernel's per-PE attention intermediates, un-tile them to "
                         "head-space, and compare stage-by-stage to the trusted reference (localizes "
                         "a per-PE/tiling/kernel bug: first stage to diverge names it)")
    ap.add_argument("--dump-dir", default="dump_intermediates",
                    help="dir to save raw grids + reconstructs + references as .npy (for forensics)")
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    P           = cfg["P"]
    bsz         = cfg["bsz"]
    dim         = cfg["dim"]
    n_heads     = cfg["n_heads"]
    n_kv_heads  = cfg["n_kv_heads"]
    head_dim    = cfg["head_dim"]
    max_seq_len = cfg["max_seq_len"]
    prefill_len = cfg["prefill_len"]
    ffn_dim     = cfg["ffn_dim"]

    dim_p_pe         = dim // P
    kv_dim           = n_kv_heads * head_dim
    kv_dim_p_pe      = kv_dim // P
    ffn_dim_p_pe     = ffn_dim // P
    gqa_group_size   = n_heads // n_kv_heads
    pes_p_kv_head    = P // n_kv_heads
    max_seq_len_p_pe = max_seq_len // P
    prefill_len_p_pe = prefill_len // P
    _dim_p_pe        = dim_p_pe if (dim_p_pe % 2 == 0) else dim_p_pe - 1

    assert bsz == 1, "verify harness assumes bsz=1 (one decode token)"

    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    # ── Oracle ────────────────────────────────────────────────────────────────
    if args.oracle_dir:
        oracle_dir = args.oracle_dir
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        oracle_dir = os.path.join(repo_root, "pytorch", "oracle_decode")
    def load_oracle(fn):
        return np.load(os.path.join(oracle_dir, fn))
    resid_pre  = load_oracle("resid_pre_block0.npy").astype(np.float32)    # [seq, dim]
    resid_mid  = load_oracle("resid_mid_block0.npy").astype(np.float32)
    resid_post = load_oracle("resid_post_block0.npy").astype(np.float32)
    k_cache    = load_oracle("k_cache_block0.npy").astype(np.float16)      # [seq, n_kv_heads, head_dim] POST-RoPE
    v_cache    = load_oracle("v_cache_block0.npy").astype(np.float16)
    seq = resid_pre.shape[0]
    assert seq > prefill_len, (f"oracle seq={seq} <= prefill_len={prefill_len}: need a longer prompt "
                               f"(>= {prefill_len + 1} tokens)")
    print(f"Host: P={P} dim={dim} n_heads={n_heads} n_kv_heads={n_kv_heads} head_dim={head_dim} "
          f"dim_p_pe={dim_p_pe} kv_dim_p_pe={kv_dim_p_pe} gqa_group_size={gqa_group_size} pes_p_kv_head={pes_p_kv_head}")
    print(f"      oracle seq={seq}, decode position = prefill_len = {prefill_len}")

    # ── Real weights ──────────────────────────────────────────────────────────
    w = load_block0_weights(head_dim, n_heads, n_kv_heads)

    # Attention-only "probe" mode: if the config's ffn_dim is smaller than the real one, the
    # FFN weights are sliced to fit (resid_mid is captured BEFORE the FFN, so it's unaffected —
    # this lets P=32 validate the attention path without the real FFN tile overflowing i16).
    ffn_real = w["up"].shape[1]
    attn_only = ffn_dim < ffn_real
    if attn_only:
        print(f"      [PROBE] ffn_dim={ffn_dim} < real {ffn_real}: attention-only — resid_post NOT validated")
        w["up"]   = w["up"][:, :ffn_dim]
        w["gate"] = w["gate"][:, :ffn_dim]
        w["down"] = w["down"][:ffn_dim, :]

    # Option-C GQA offline permutation on Q (columns) and O (rows); K/V unpermuted.
    W_Q_perm = np.zeros((dim, dim), dtype=np.float16)
    W_O_perm = np.zeros((dim, dim), dtype=np.float16)
    for h in range(n_heads):
        kv_head = h // gqa_group_size
        g = h % gqa_group_size
        for s in range(pes_p_kv_head):
            old_col = h * head_dim + s * kv_dim_p_pe
            new_col = kv_head * pes_p_kv_head * dim_p_pe + s * dim_p_pe + g * kv_dim_p_pe
            W_Q_perm[:, new_col:new_col + kv_dim_p_pe] = w["q"][:, old_col:old_col + kv_dim_p_pe]
            W_O_perm[new_col:new_col + kv_dim_p_pe, :] = w["o"][old_col:old_col + kv_dim_p_pe, :]

    def tile_weight_row(W, rows, cols):
        return W.reshape(P, rows, P, cols).transpose(0, 2, 1, 3).reshape(P, P, rows * cols)

    Q_tile  = tile_weight_row(W_Q_perm,     dim_p_pe, dim_p_pe)
    K_tile  = tile_weight_row(w["k"],       dim_p_pe, kv_dim_p_pe)
    V_tile  = tile_weight_row(w["v"],       dim_p_pe, kv_dim_p_pe)
    O_tile  = tile_weight_row(W_O_perm,     dim_p_pe, dim_p_pe)
    UP_tile = tile_weight_row(w["up"],      dim_p_pe, ffn_dim_p_pe)
    GT_tile = tile_weight_row(w["gate"],    dim_p_pe, ffn_dim_p_pe)
    DN_tile = tile_weight_row(w["down"],    ffn_dim_p_pe, dim_p_pe)

    # Norm weights: dim-block by py, replicated across px (matches X's tiling).
    tensor_W  = np.tile(w["norm_pre"].reshape(P, dim_p_pe),  reps=(1, P))
    tensor_W2 = np.tile(w["norm_post"].reshape(P, dim_p_pe), reps=(1, P))

    # ── Real RoPE freqs at the decode position (see derivation below) ──────────
    # PE column px holds Q output cols [px*dim_p_pe:(px+1)*dim_p_pe] in W_Q_perm space, which
    # = kv_head=px//pes_p_kv_head, s=px%pes_p_kv_head, g=0..gqa_group_size-1. Within a g-block the
    # kernel's adjacent pair (2m,2m+1) is HF rotate-half pair i = s*(kv_dim_p_pe/2)+m. So each PE's
    # freq buffer is base[s-segment] tiled gqa_group_size times (same for Q's g-blocks and K).
    pos = prefill_len
    inv_freq = THETA ** (-2.0 * np.arange(head_dim // 2) / head_dim)   # [head_dim//2]
    base_cos = np.cos(pos * inv_freq).astype(np.float16)              # base_cos[i] for HF pair i
    base_sin = np.sin(pos * inv_freq).astype(np.float16)
    half_per_s = kv_dim_p_pe // 2
    pe_freqs_cos = np.zeros((P, _dim_p_pe // 2), dtype=np.float16)
    pe_freqs_sin = np.zeros((P, _dim_p_pe // 2), dtype=np.float16)
    for px in range(P):
        s = px % pes_p_kv_head
        seg_c = base_cos[s * half_per_s:(s + 1) * half_per_s]
        seg_s = base_sin[s * half_per_s:(s + 1) * half_per_s]
        pe_freqs_cos[px, :] = np.tile(seg_c, gqa_group_size)
        pe_freqs_sin[px, :] = np.tile(seg_s, gqa_group_size)
    tensor_freqs_cos = np.tile(pe_freqs_cos.ravel(), (P, 1))
    tensor_freqs_sin = np.tile(pe_freqs_sin.ravel(), (P, 1))

    # ── KV cache seed from oracle (positions 0..prefill_len-1; kernel appends position prefill_len) ──
    # k_cache is HF post-RoPE; permute its head_dim to kernel pair order (same rope_perm as weights).
    kperm = rope_perm_idx(head_dim)
    k_seed = k_cache[:prefill_len][:, :, kperm]          # [prefill_len, n_kv_heads, head_dim]
    tensor_XKCache = np.zeros((bsz, kv_dim, max_seq_len), dtype=np.float16)
    tensor_XVCache = np.zeros((bsz, max_seq_len, kv_dim), dtype=np.float16)
    for t in range(prefill_len):
        tensor_XKCache[0, :, t] = k_seed[t].reshape(kv_dim)         # kv_dim = kv_head*head_dim + d
        tensor_XVCache[0, t, :] = v_cache[t].reshape(kv_dim)
    KCache_tile = tile_kcache_interleaved(tensor_XKCache, P, bsz, kv_dim_p_pe, max_seq_len_p_pe, prefill_len_p_pe)
    VCache_tile = tile_vcache_interleaved(tensor_XVCache, P, bsz, kv_dim_p_pe, max_seq_len_p_pe, prefill_len_p_pe)

    # ── X = the decode token's input residual, dim-block by py, replicated across px ──
    X_raw = resid_pre[prefill_len].astype(np.float16)                 # [dim]
    tensor_X = np.tile(X_raw.reshape(P, dim_p_pe), reps=(1, P))

    # ── Runner (client API: cloud artifact; simulator=False -> real WSE-3, like Prefill) ──
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    with open(os.path.join(OUT_PATH, f"artifact_{cfg_name}.json"), encoding="utf-8") as f:
        artifact_id = json.load(f)["artifact_id"]

    with SdkRuntime(artifact_id, simulator=args.simulator) as runner:
        def h2d(name, flat_arr, count_per_pe):
            runner.memcpy_h2d(runner.get_id(name), cast_tensor_u32(flat_arr.ravel()), 0, 0, P, P, count_per_pe,
                              streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

        h2d("X",           tensor_X,         bsz * dim_p_pe)
        h2d("W",           tensor_W,         dim_p_pe)
        h2d("W2",          tensor_W2,        dim_p_pe)
        h2d("Q_weight",    Q_tile,           dim_p_pe * dim_p_pe)
        h2d("K_weight",    K_tile,           dim_p_pe * kv_dim_p_pe)
        h2d("V_weight",    V_tile,           dim_p_pe * kv_dim_p_pe)
        h2d("freqs_sin",   tensor_freqs_sin, _dim_p_pe // 2)
        h2d("freqs_cos",   tensor_freqs_cos, _dim_p_pe // 2)
        h2d("XKCache",     KCache_tile,      bsz * kv_dim_p_pe * max_seq_len_p_pe)
        h2d("XVCache",     VCache_tile,      bsz * max_seq_len_p_pe * kv_dim_p_pe)
        h2d("O_weight",    O_tile,           dim_p_pe * dim_p_pe)
        h2d("UP_weight",   UP_tile,          dim_p_pe * ffn_dim_p_pe)
        h2d("GATE_weight", GT_tile,          dim_p_pe * ffn_dim_p_pe)
        h2d("DOWN_weight", DN_tile,          ffn_dim_p_pe * dim_p_pe)

        runner.launch("init_task", nonblock=False)
        runner.launch("decode_host", np.int16(1), np.int16(0), nonblock=False)   # 1 decode step

        pre_grid  = d2h(runner, runner.get_id("resid_pre"),  P, bsz, dim_p_pe, io_dtype, memcpy_order)
        mid_grid  = d2h(runner, runner.get_id("resid_mid"),  P, bsz, dim_p_pe, io_dtype, memcpy_order)
        post_grid = d2h(runner, runner.get_id("resid_post"), P, bsz, dim_p_pe, io_dtype, memcpy_order)

        # Per-PE attention intermediates (only when localizing). Reduced/broadcast buffers
        # (QKV_*, output_tile) are read at any py; score buffers keep per-py token rows.
        dump = {}
        if args.dump_intermediates:
            qkv_wide = dim_p_pe + 2 * kv_dim_p_pe                 # Q(dim_p_pe) K/V(kv_dim_p_pe each)
            score_wide = gqa_group_size * max_seq_len_p_pe
            dump["QKV_post_reduce"] = d2h(runner, runner.get_id("QKV_post_reduce"), P, bsz, qkv_wide, io_dtype, memcpy_order)
            dump["QKV_tile"]        = d2h(runner, runner.get_id("QKV_tile"),        P, bsz, qkv_wide, io_dtype, memcpy_order)
            dump["score_post_reduce"] = d2h(runner, runner.get_id("score_post_reduce"), P, bsz, score_wide, io_dtype, memcpy_order)
            dump["score"]           = d2h(runner, runner.get_id("score"),           P, bsz, score_wide, io_dtype, memcpy_order)
            dump["output_tile"]     = d2h(runner, runner.get_id("output_tile"),     P, bsz, dim_p_pe, io_dtype, memcpy_order)

    # Residual stream is dim-block by py, replicated across px -> gather py, use px=0.
    def reconstruct(grid):
        out = np.zeros(P * dim_p_pe, dtype=np.float32)
        for py in range(P):
            out[py * dim_p_pe:(py + 1) * dim_p_pe] = grid[py, 0, :].astype(np.float32)
        return out

    kernel_pre  = reconstruct(pre_grid)
    kernel_mid  = reconstruct(mid_grid)
    kernel_post = reconstruct(post_grid)
    o_pre  = resid_pre[prefill_len]
    o_mid  = resid_mid[prefill_len]
    o_post = resid_post[prefill_len]

    # Hand the DEVICE resid_mid to the FFN stage (launch_layer_verify chains this .npy). fp16 to
    # preserve exactly what the device produced (no host re-rounding).
    np.save(args.save_zmid, kernel_mid.astype(np.float16))

    sep(f"transformer_lens validation — decode step at position {prefill_len}")
    all_ok = True
    all_ok &= report("resid_pre (Z_pre echo)", kernel_pre, o_pre)
    all_ok &= report("resid_mid",  kernel_mid,  o_mid)
    all_ok &= report_contrib("  attn contribution", kernel_mid,  o_mid,  o_pre)
    if attn_only:
        print("  (resid_post skipped — attention stage; FFN runs as a separate P=128 launch)")
    else:
        all_ok &= report("resid_post", kernel_post, o_post)
        all_ok &= report_contrib("  ffn contribution",  kernel_post, o_post, o_mid)
    print()
    print(f"  device Z_mid saved -> {args.save_zmid}")
    print(f"  Overall: {'ALL PASS ✓' if all_ok else 'SOME FAILURES — check above'}")

    # ── Localization: un-tile the kernel's attention intermediates to head-space & compare ──
    # verify_numpy already proved the head-space CONVENTIONS are right, so a device failure is a
    # per-PE tiling / freq-layout / kernel bug. Reconstruct each exported intermediate back to
    # head-space and compare in PIPELINE ORDER — the first stage to diverge is where it breaks.
    if args.dump_intermediates:
        sep(f"per-PE intermediate localization — decode step at position {prefill_len}")
        pos = prefill_len
        iter_num = pos // P + 1                        # cache slots in use (batch-outer score stride)
        kperm = rope_perm_idx(head_dim)                # kernel adjacent-pair col k -> HF col kperm[k]

        # Trusted head-space reference (raw weights + oracle cache; identical math to verify_numpy).
        rw = load_raw_attn_weights()
        x0 = resid_pre[pos].astype(np.float64)
        xn = x0 / np.sqrt(np.mean(x0 ** 2) + 1e-5) * rw["norm"].astype(np.float64)
        Q_hf = (xn @ rw["q"].T.astype(np.float64)).reshape(n_heads, head_dim)
        K_hf = (xn @ rw["k"].T.astype(np.float64)).reshape(n_kv_heads, head_dim)
        V_hf = (xn @ rw["v"].T.astype(np.float64)).reshape(n_kv_heads, head_dim)
        Qr_hf, Kr_hf = rope_hf(Q_hf, pos), rope_hf(K_hf, pos)
        K_all = k_cache[:pos + 1].astype(np.float64)   # oracle post-rope K [pos+1, n_kv, head_dim]
        V_all = v_cache[:pos + 1].astype(np.float64)
        scale = 1.0 / np.sqrt(head_dim)
        logits_ref = np.zeros((n_heads, pos + 1)); attn_ref = np.zeros((n_heads, pos + 1))
        out_ref = np.zeros((n_heads, head_dim))
        for h in range(n_heads):
            kv = h // gqa_group_size
            s = (Qr_hf[h] @ K_all[:, kv].T) * scale
            logits_ref[h] = s
            e = np.exp(s - s.max()); attn_ref[h] = e / e.sum()
            out_ref[h] = attn_ref[h] @ V_all[:, kv]

        # Reconstruct = invert the exact host tiling (Q via Option-C W_Q_perm; K/V plain row-tile;
        # score via the token interleave py=t%P, slot=t//P; output via Option-C group-inner layout).
        def gather_cols(grid, offset, width):          # reduced/broadcast buffer: any py (use 0)
            v = np.zeros(P * width, dtype=np.float32)
            for px in range(P):
                v[px * width:(px + 1) * width] = grid[0, px, offset:offset + width].astype(np.float32)
            return v

        def invert_optionC(q_perm):                    # W_Q_perm output-col space -> kernel-pair [dim]
            qk = np.zeros(dim, dtype=np.float32)
            for h in range(n_heads):
                kv_head = h // gqa_group_size; g = h % gqa_group_size
                for s in range(pes_p_kv_head):
                    old = h * head_dim + s * kv_dim_p_pe
                    new = kv_head * pes_p_kv_head * dim_p_pe + s * dim_p_pe + g * kv_dim_p_pe
                    qk[old:old + kv_dim_p_pe] = q_perm[new:new + kv_dim_p_pe]
            return qk

        def kern_to_hf(vec, n):                         # kernel adjacent-pair order -> HF order
            v = vec.reshape(n, head_dim); out = np.zeros_like(v); out[:, kperm] = v; return out

        def recon_Q(grid): return kern_to_hf(invert_optionC(gather_cols(grid, 0, dim_p_pe)), n_heads)
        def recon_K(grid): return kern_to_hf(gather_cols(grid, dim_p_pe, kv_dim_p_pe), n_kv_heads)
        def recon_V(grid): return gather_cols(grid, dim_p_pe + kv_dim_p_pe, kv_dim_p_pe).reshape(n_kv_heads, head_dim)

        def recon_output(grid):                         # Option-C group-inner: PE px=(kv_head,s), block g
            out = np.zeros((n_heads, head_dim), dtype=np.float32)
            for px in range(P):
                kv_head = px // pes_p_kv_head; s = px % pes_p_kv_head
                for g in range(gqa_group_size):
                    h = kv_head * gqa_group_size + g
                    out[h, s * kv_dim_p_pe:(s + 1) * kv_dim_p_pe] = \
                        grid[0, px, g * kv_dim_p_pe:(g + 1) * kv_dim_p_pe].astype(np.float32)
            return out

        def recon_scores(grid):                         # -> [n_heads, pos+1]; token t on py=t%P, slot=t//P
            out = np.zeros((n_heads, pos + 1), dtype=np.float32)
            for h in range(n_heads):
                kv_head = h // gqa_group_size; g = h % gqa_group_size
                px = kv_head * pes_p_kv_head            # score broadcast across kv-head's px range
                for t in range(pos + 1):
                    out[h, t] = grid[t % P, px, g * iter_num + (t // P)].astype(np.float32)
            return out

        d_ok = True
        d_ok &= report("QKV_post_reduce Q (proj)",   recon_Q(dump["QKV_post_reduce"]), Q_hf)
        d_ok &= report("QKV_post_reduce K (proj)",   recon_K(dump["QKV_post_reduce"]), K_hf)
        d_ok &= report("QKV_post_reduce V (proj)",   recon_V(dump["QKV_post_reduce"]), V_hf)
        d_ok &= report("QKV_tile Q (post-rope)",     recon_Q(dump["QKV_tile"]),        Qr_hf)
        d_ok &= report("QKV_tile K (post-rope)",     recon_K(dump["QKV_tile"]),        Kr_hf)
        d_ok &= report("score_post_reduce (logits)", recon_scores(dump["score_post_reduce"]), logits_ref)
        d_ok &= report("score (softmax)",            recon_scores(dump["score"]),      attn_ref)
        d_ok &= report("output_tile (attn·V)",       recon_output(dump["output_tile"]), out_ref)

        os.makedirs(args.dump_dir, exist_ok=True)
        for k, v in dump.items():
            np.save(os.path.join(args.dump_dir, f"raw_{k}.npy"), v)
        for nm, arr in {"Q": Q_hf, "K": K_hf, "V": V_hf, "Qr": Qr_hf, "Kr": Kr_hf,
                        "logits": logits_ref, "attn": attn_ref, "output": out_ref}.items():
            np.save(os.path.join(args.dump_dir, f"ref_{nm}.npy"), arr)
        print(f"\n  raw grids + references saved -> {args.dump_dir}/")
        print("  Localization: " + ("all attention stages reconstruct correctly — bug is downstream "
              "(o_proj W_O_perm tiling / residual add)" if d_ok else
              "first FAIL above = the diverging stage (that host-prep tiling or kernel step is the bug)"))


if __name__ == "__main__":
    main()
