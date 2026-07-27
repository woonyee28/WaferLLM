"""Numpy self-check of launch_verify.py's P=64 RoPE freq layout vs HF rotate-half.

Pure host-side math (no PEs/sim): does the kernel's rope pipeline
  rope_perm(weights) -> Option-C W_Q_perm -> per-PE tile -> per-PE freqs -> adjacent-pair rope
reproduce HuggingFace Llama rotate-half RoPE at real Llama-3-8B dims? This isolates the ONE
piece launch_verify derives rather than validates: the per-PE freq segmentation.
"""
import numpy as np

# Real Llama-3-8B block-0 dims + P=64 mesh
P, n_heads, n_kv_heads, head_dim = 64, 32, 8, 128
dim = n_heads * head_dim                      # 4096
pos = 64                                      # decode token position
theta = 500000.0

dim_p_pe       = dim // P                      # 64
kv_dim_p_pe    = (n_kv_heads * head_dim) // P  # 16
gqa_group_size = n_heads // n_kv_heads         # 4
pes_p_kv_head  = P // n_kv_heads               # 8
half           = head_dim // 2                 # 64

rng = np.random.default_rng(0)
x        = rng.standard_normal(dim).astype(np.float64)
q_weight = rng.standard_normal((dim, dim)).astype(np.float64)   # [in, out]; WaferLLM computes x@W

# ── HF reference: q = x@W -> [n_heads, head_dim] -> rotate-half rope ──────────
def hf_rope(q):
    f = pos * theta ** (-2.0 * np.arange(half) / head_dim)      # [half]
    c, s = np.cos(f), np.sin(f)
    q1, q2 = q[:, :half], q[:, half:]
    out = np.empty_like(q)
    out[:, :half] = q1 * c - q2 * s
    out[:, half:] = q2 * c + q1 * s
    return out

q_hf = hf_rope((x @ q_weight).reshape(n_heads, head_dim))

# ── Kernel path ──────────────────────────────────────────────────────────────
# rope_perm within each head: kernel col 2i <- HF col i+half, 2i+1 <- HF col i
p = np.empty(head_dim, dtype=np.int64); p[0::2] = np.arange(half) + half; p[1::2] = np.arange(half)
full = np.concatenate([h * head_dim + p for h in range(n_heads)])
q_rp = q_weight[:, full]

# Option-C W_Q_perm (column permutation)
WQ = np.zeros((dim, dim), dtype=np.float64)
for h in range(n_heads):
    kvh, g = h // gqa_group_size, h % gqa_group_size
    for s in range(pes_p_kv_head):
        oc = h * head_dim + s * kv_dim_p_pe
        nc = kvh * pes_p_kv_head * dim_p_pe + s * dim_p_pe + g * kv_dim_p_pe
        WQ[:, nc:nc + kv_dim_p_pe] = q_rp[:, oc:oc + kv_dim_p_pe]

# per-PE freqs (launch_verify's derived layout)
inv = theta ** (-2.0 * np.arange(half) / head_dim)
base_cos, base_sin = np.cos(pos * inv), np.sin(pos * inv)
hps = kv_dim_p_pe // 2
fcos = np.zeros((P, dim_p_pe // 2)); fsin = np.zeros((P, dim_p_pe // 2))
for px in range(P):
    s = px % pes_p_kv_head
    fcos[px] = np.tile(base_cos[s * hps:(s + 1) * hps], gqa_group_size)
    fsin[px] = np.tile(base_sin[s * hps:(s + 1) * hps], gqa_group_size)

# per-PE Q block + kernel adjacent-pair rope
qfull = x @ WQ                                  # [dim] in permuted output space
q_perm_roped = np.zeros(dim, dtype=np.float64)
for px in range(P):
    blk = qfull[px * dim_p_pe:(px + 1) * dim_p_pe].copy()
    for i in range(dim_p_pe // 2):
        e, o = blk[2 * i], blk[2 * i + 1]
        c, sn = fcos[px, i], fsin[px, i]
        blk[2 * i]     = o * c - e * sn
        blk[2 * i + 1] = e * c + o * sn
    q_perm_roped[px * dim_p_pe:(px + 1) * dim_p_pe] = blk

# invert Option-C -> per-head kernel-pair order
q_khead = np.zeros(n_heads * head_dim, dtype=np.float64)
for h in range(n_heads):
    kvh, g = h // gqa_group_size, h % gqa_group_size
    for s in range(pes_p_kv_head):
        oc = h * head_dim + s * kv_dim_p_pe
        nc = kvh * pes_p_kv_head * dim_p_pe + s * dim_p_pe + g * kv_dim_p_pe
        q_khead[oc:oc + kv_dim_p_pe] = q_perm_roped[nc:nc + kv_dim_p_pe]
q_khead = q_khead.reshape(n_heads, head_dim)

# kernel pair order: [2i]=HF_roped[i], [2i+1]=HF_roped[i+half] -> back to HF order
q_kernel_hf = np.zeros((n_heads, head_dim), dtype=np.float64)
q_kernel_hf[:, :half] = q_khead[:, 0::2]
q_kernel_hf[:, half:] = q_khead[:, 1::2]

err = np.abs(q_kernel_hf - q_hf).max()
cos = float(np.dot(q_kernel_hf.ravel(), q_hf.ravel()) /
            (np.linalg.norm(q_kernel_hf) * np.linalg.norm(q_hf)))
print(f"P={P} head_dim={head_dim} gqa_group_size={gqa_group_size} pes_p_kv_head={pes_p_kv_head}")
print(f"max_abs_err = {err:.3e}")
print(f"cosine      = {cos:.10f}   -> {'PASS (freq layout reproduces HF rope)' if cos > 0.99999 else 'FAIL (freq layout WRONG)'}")
