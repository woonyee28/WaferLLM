"""verify_numpy.py — pure-numpy ground-truth decode attention vs transformer_lens (NO device, NO sim).

Reproduces the block-0 decode step at position `prefill_len` in plain numpy using the REAL Llama-3-8B
weights + the transformer_lens oracle cache, and compares STAGE BY STAGE to the oracle. This isolates
whether the *reference pipeline* (weights + rope + oracle KV cache + GQA + o_proj) reproduces
transformer_lens — the ground truth the kernel must match. It uses standard HF/head-space math (NO
per-PE tiling / Option-C perm), so:

  * all stages PASS  -> weights+cache+conventions are correct; a device failure is then a KERNEL /
                        per-PE-layout bug (freq tiling, cache interleave, GQA reduce) -> read the
                        kernel's exported intermediates next.
  * a stage FAILS    -> that convention is wrong (norm / rope theta / GQA mapping / scale / o_proj).

Runs on the usernode (numpy + torch + safetensors + the HF-cached weights + pytorch/oracle_decode/):
    python verify_numpy.py [--pos 64]
"""
import os
import glob
import argparse
import numpy as np


def load_attn_weights():
    """Raw HF block-0 attention weights (NO rope_perm / NO transpose) as fp32 numpy."""
    import torch
    from safetensors import safe_open
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_dir = os.path.join(hf_home, "hub", "models--meta-llama--Meta-Llama-3-8B")
    prefix = "model.layers.0."
    want = {
        "q": "self_attn.q_proj.weight", "k": "self_attn.k_proj.weight",
        "v": "self_attn.v_proj.weight", "o": "self_attn.o_proj.weight",
        "norm": "input_layernorm.weight",
    }
    raw = {}
    for shard in sorted(glob.glob(os.path.join(model_dir, "**", "*.safetensors"), recursive=True)):
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                if name.startswith(prefix):
                    raw[name] = f.get_tensor(name).to(torch.float32).cpu().numpy()
    return {k: raw[prefix + s] for k, s in want.items()}


def cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def report(tag, got, ref):
    c = cos(got, ref)
    e = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    print(f"  [{'PASS' if c >= 0.999 else 'FAIL'}] {tag:34s} cosine={c:.6f}  max_abs={e.max():.3e}  mean_abs={e.mean():.3e}")
    return c >= 0.999


def rope_hf(x, pos, theta=500000.0):
    """HF rotate-half rope on x[..., head_dim] at scalar position `pos`."""
    hd = x.shape[-1]; half = hd // 2
    inv = theta ** (-2.0 * np.arange(half) / hd)
    ang = pos * inv
    cos_f = np.concatenate([np.cos(ang), np.cos(ang)])
    sin_f = np.concatenate([np.sin(ang), np.sin(ang)])
    x1, x2 = x[..., :half], x[..., half:]
    rot = np.concatenate([-x2, x1], axis=-1)
    return x * cos_f + rot * sin_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=64, help="decode position (= attention prefill_len)")
    ap.add_argument("--oracle-dir", default=None)
    args = ap.parse_args()

    od = args.oracle_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "pytorch", "oracle_decode")
    L = lambda n: np.load(os.path.join(od, n)).astype(np.float64)
    resid_pre, resid_mid = L("resid_pre_block0.npy"), L("resid_mid_block0.npy")
    k_cache, v_cache = L("k_cache_block0.npy"), L("v_cache_block0.npy")   # [seq, n_kv, hd] post-rope K, V
    attn_pat = L("attn_pattern_block0.npy")                                # [n_heads, seq, seq]

    seq, n_kv, hd = k_cache.shape
    n_heads = attn_pat.shape[0]
    dim = resid_pre.shape[1]
    group = n_heads // n_kv
    pos = args.pos
    assert seq > pos, f"oracle seq={seq} <= pos={pos}"
    print(f"dims: dim={dim} n_heads={n_heads} n_kv={n_kv} head_dim={hd} group={group} pos={pos} seq={seq}")

    # Weight-free oracle sanity: confirm k_cache is genuine HF post-rope (θ=500000) before trusting it.
    k_pre = L("k_pre_rope_block0.npy")
    k_roped = np.stack([rope_hf(k_pre[t], t) for t in range(seq)])
    print(f"oracle sanity: k_cache vs rope(k_pre_rope) cosine={cos(k_roped, k_cache):.6f} "
          f"(should be ~1.0; else oracle K hook is wrong)")

    w = load_attn_weights()
    x = resid_pre[pos]                                        # [dim]

    # RMSNorm (Llama: eps=1e-5, mean over full dim)
    xn = x / np.sqrt(np.mean(x**2) + 1e-5) * w["norm"]

    # Projections (HF stores [out,in]; logical y = x @ W.T)
    Q = (xn @ w["q"].T).reshape(n_heads, hd)
    Knew = (xn @ w["k"].T).reshape(n_kv, hd)
    Vnew = (xn @ w["v"].T).reshape(n_kv, hd)

    # RoPE at this position on Q and the new K
    Qr = rope_hf(Q, pos)
    Knew_r = rope_hf(Knew, pos)

    print("\ntransformer_lens stage-by-stage:")
    all_ok = True
    # (a) K projection+rope: our new-token K must equal the oracle's cached K at `pos`
    all_ok &= report("K[pos] proj+rope vs oracle", Knew_r, k_cache[pos])
    all_ok &= report("V[pos] proj vs oracle",       Vnew,   v_cache[pos])

    # (b) scores -> softmax over the causal window [0..pos], compare to oracle attn_pattern
    K_all = k_cache[:pos + 1]           # [pos+1, n_kv, hd] (oracle post-rope; includes pos)
    V_all = v_cache[:pos + 1]
    scale = 1.0 / np.sqrt(hd)
    attn = np.zeros((n_heads, pos + 1))
    for h in range(n_heads):
        s = (Qr[h] @ K_all[:, h // group].T) * scale        # [pos+1]
        s -= s.max()
        e = np.exp(s)
        attn[h] = e / e.sum()
    all_ok &= report("softmax attn vs attn_pattern", attn, attn_pat[:, pos, :pos + 1])

    # (c) output -> o_proj -> residual add, compare to oracle resid_mid
    out = np.zeros((n_heads, hd))
    for h in range(n_heads):
        out[h] = attn[h] @ V_all[:, h // group]
    o = out.reshape(dim) @ w["o"].T
    resid_mid_model = x + o
    all_ok &= report("resid_mid (pre + o_proj(attn))", resid_mid_model, resid_mid[pos])

    print(f"\n  {'ALL STAGES PASS -> reference pipeline is correct; a device failure is a KERNEL/tiling bug.' if all_ok else 'A STAGE FAILED -> that convention is the bug (see first FAIL above).'}")


if __name__ == "__main__":
    main()
