import os
import numpy as np
import argparse
import json
import struct

from cerebras.sdk.client import SdkRuntime, sdk_utils
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder

out_path = "compile_out"

# wyn: FFN-only launcher for the disaggregated layer. Runs the FFN program (compiled with
# ffn_only=1) at P=256. Input is Llama block-0 resid_mid; output is resid_post = resid_mid +
# mlp(norm_post(resid_mid)). Loads only norm_post + gate/up/down (no attention weights).

def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])

def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)

def cast_tensor_u32(tensor):
    return np.uint32(tensor.view(np.uint16))

def report_match(name, got, ref):
    # wyn: fp16 kernel vs reference. Cosine similarity is the scale-robust "same function?" test;
    # max relative error is judged only on significant elements (|ref|>0.5) to avoid tiny-denominator
    # blow-ups. PASS on cosine >= 0.999 (a correct fp16 layer sits well above this).
    got = got.astype(np.float32); ref = ref.astype(np.float32)
    abs_err = np.abs(got - ref)
    sig = np.abs(ref) > 0.5
    max_rel = float((abs_err[sig] / np.abs(ref)[sig]).max()) if sig.any() else 0.0
    g, r = got.ravel(), ref.ravel()
    cos = float(np.dot(g, r) / (np.linalg.norm(g) * np.linalg.norm(r)))
    ok = cos >= 0.999
    print(f"[{name}] max_abs={abs_err.max():.3e} mean_abs={abs_err.mean():.3e} "
          f"max_rel(|ref|>0.5)={max_rel:.2%} cos={cos:.6f} -> {'PASS' if ok else 'FAIL'}")
    return ok

def untile_flat_1d(input_flat_1d, P, seq_len_p_pe, dim_p_pe):
    a = input_flat_1d.reshape(P, P, dim_p_pe, seq_len_p_pe)
    a = a.transpose(0, 3, 1, 2)
    return a.reshape(seq_len_p_pe * P, dim_p_pe * P)

def load_ffn_weights():
    import glob, torch
    from safetensors import safe_open
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_dir = os.path.join(hf_home, "hub", "models--meta-llama--Meta-Llama-3-8B")
    prefix = "model.layers.0."
    want = {
        "gate": "mlp.gate_proj.weight",
        "up":   "mlp.up_proj.weight",
        "down": "mlp.down_proj.weight",
        "norm_post": "post_attention_layernorm.weight",
    }
    raw = {}
    for shard in sorted(glob.glob(os.path.join(model_dir, "**", "*.safetensors"), recursive=True)):
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                if name.startswith(prefix):
                    raw[name] = f.get_tensor(name).to(torch.float32).cpu().numpy().astype(np.float16)
    out = {}
    for k, suf in want.items():
        w = raw[prefix + suf]
        out[k] = w if k == "norm_post" else w.T   # HF [out,in] x@W.T -> [in,out] for x@W
    return out

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
        self.P = 256
        self.dim = 4096
        self.n_heads = 16
        self.n_kv_heads = 4
        self.head_dim = 256
        self.seq_len = 256
        self.ffn_dim = 14336

def parse_args():
    parser = argparse.ArgumentParser(description="FFN-only Prefill on WSE-3")
    parser.add_argument("--config", default="config.json", type=str, help="Config file")
    args = parser.parse_args()
    return args

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

    dim_p_pe = dim // P
    seq_len_p_pe = seq_len // P
    ffn_dim_p_pe = ffn_dim // P

    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    weights = load_ffn_weights()

    # wyn: input = Llama block-0 resid_mid (oracle), so the FFN is validated in isolation.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    resid_dir = os.path.join(repo_root, "pytorch")
    resid_mid = np.load(os.path.join(resid_dir, "resid_mid_block0.npy")).astype(np.float16)
    n_tok = resid_mid.shape[0]
    assert n_tok <= seq_len, f"resid_mid has {n_tok} tokens > seq_len {seq_len}"
    tensor_X = np.zeros((seq_len, dim), dtype=np.float16)
    tensor_X[:n_tok] = resid_mid

    W2 = weights["norm_post"].reshape(1, dim)
    tensor_W2 = np.tile(W2.reshape(P, dim_p_pe), reps=(1, P))

    tensor_gate_weight = weights["gate"]
    tensor_up_weight = weights["up"]
    tensor_down_weight = weights["down"]

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

    tensor_up_weight_shifted = np.zeros((dim, ffn_dim)).astype(np.float16)
    tensor_gate_weight_shifted = np.zeros((dim, ffn_dim)).astype(np.float16)
    tensor_down_weight_shifted = np.zeros((ffn_dim, dim)).astype(np.float16)
    for i in range(P):
        for j in range(P):
            t = ind[i, j]
            tensor_up_weight_shifted[i*dim_p_pe:(i+1)*dim_p_pe, j*ffn_dim_p_pe:(j+1)*ffn_dim_p_pe] = tensor_up_weight[t*dim_p_pe:(t+1)*dim_p_pe, j*ffn_dim_p_pe:(j+1)*ffn_dim_p_pe]
            tensor_gate_weight_shifted[i*dim_p_pe:(i+1)*dim_p_pe, j*ffn_dim_p_pe:(j+1)*ffn_dim_p_pe] = tensor_gate_weight[t*dim_p_pe:(t+1)*dim_p_pe, j*ffn_dim_p_pe:(j+1)*ffn_dim_p_pe]
            tensor_down_weight_shifted[i*ffn_dim_p_pe:(i+1)*ffn_dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe] = tensor_down_weight[t*ffn_dim_p_pe:(t+1)*ffn_dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe]

    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    with open(f"{out_path}/artifact_{cfg_name}.json", "r", encoding="utf-8") as f:
        artifact_id = json.load(f)["artifact_id"]

    with SdkRuntime(artifact_id, simulator=False) as runner:
        sym_X = runner.get_id("X")
        sym_W2 = runner.get_id("W2")
        sym_UP_weight = runner.get_id("UP_weight")
        sym_GATE_weight = runner.get_id("GATE_weight")
        sym_DOWN_weight = runner.get_id("DOWN_weight")
        symbol_time_memcpy = runner.get_id("time_memcpy")
        symbol_time_ref = runner.get_id("time_ref")

        Xc = tensor_X.reshape(P, seq_len_p_pe, P, dim_p_pe).transpose(0, 2, 3, 1).reshape(P, P, seq_len_p_pe * dim_p_pe)
        runner.memcpy_h2d(sym_X, cast_tensor_u32(Xc.ravel()), 0, 0, P, P, seq_len_p_pe * dim_p_pe,
                          streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

        runner.memcpy_h2d(sym_W2, cast_tensor_u32(tensor_W2.ravel()), 0, 0, P, P, dim_p_pe,
                          streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

        UP = tensor_up_weight_shifted.reshape(P, dim_p_pe, P, ffn_dim_p_pe).transpose(0, 2, 1, 3).reshape(P, P, dim_p_pe * ffn_dim_p_pe)
        runner.memcpy_h2d(sym_UP_weight, cast_tensor_u32(UP.ravel()), 0, 0, P, P, dim_p_pe * ffn_dim_p_pe,
                          streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

        GATE = tensor_gate_weight_shifted.reshape(P, dim_p_pe, P, ffn_dim_p_pe).transpose(0, 2, 1, 3).reshape(P, P, dim_p_pe * ffn_dim_p_pe)
        runner.memcpy_h2d(sym_GATE_weight, cast_tensor_u32(GATE.ravel()), 0, 0, P, P, dim_p_pe * ffn_dim_p_pe,
                          streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

        DOWN = tensor_down_weight_shifted.reshape(P, ffn_dim_p_pe, P, dim_p_pe).transpose(0, 2, 1, 3).reshape(P, P, ffn_dim_p_pe * dim_p_pe)
        runner.memcpy_h2d(sym_DOWN_weight, cast_tensor_u32(DOWN.ravel()), 0, 0, P, P, ffn_dim_p_pe * dim_p_pe,
                          streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

        runner.launch('init_task', nonblock=False)
        runner.launch('prefill_host', np.int16(10), np.int16(100), nonblock=False)

        time_memcpy_1d_f32 = np.zeros(P*P*3, dtype=np.float32)
        runner.memcpy_d2h(time_memcpy_1d_f32, symbol_time_memcpy, 0, 0, P, P, 3, streaming=False,
                          order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
        time_ref_1d_f32 = np.zeros(P*P*2, np.float32)
        runner.memcpy_d2h(time_ref_1d_f32, symbol_time_ref, 0, 0, P, P, 2, streaming=False,
                          order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

        sym_Z = runner.get_id("Z")
        Z_1d_u32 = np.zeros(P * P * seq_len_p_pe * dim_p_pe, dtype=np.uint32)
        runner.memcpy_d2h(Z_1d_u32, sym_Z, 0, 0, P, P, seq_len_p_pe * dim_p_pe,
                          streaming=False, order=memcpy_order, data_type=io_dtype, nonblock=False)
        Z_1d = sdk_utils.memcpy_view(Z_1d_u32, np.dtype(np.float16))
        Z_layer0 = untile_flat_1d(Z_1d, P, seq_len_p_pe, dim_p_pe)

    # wyn: fp16-appropriate match report. Absolute tol is too strict for fp16 accumulation on the
    # few large (massive-activation) residual elements, so pass on cosine similarity (does the kernel
    # compute the right function?) and report max relative error on significant elements for context.
    resid_post = np.load(os.path.join(resid_dir, "resid_post_block0.npy")).astype(np.float32)
    report_match("resid_post", Z_layer0[:resid_post.shape[0]], resid_post)

    # wyn: CONTRIBUTION cosine -- the honest metric. Z_layer0 = resid_mid + mlp(...), so the full
    # cosine above rides on the resid_mid passthrough and hides FFN error: on the P=8 sim it read
    # 0.986 while the FFN contribution was 0.367 (a buffer-parity bug in z2_matmul, now fixed).
    # Strip the passthrough: compare (Z - resid_mid) against (resid_post - resid_mid).
    n = resid_post.shape[0]
    base = resid_mid.astype(np.float64)[:n]
    g = Z_layer0.astype(np.float64)[:n] - base
    r = resid_post.astype(np.float64)[:n] - base
    gf, rf = g.ravel(), r.ravel()
    cos_c = float(np.dot(gf, rf) / (np.linalg.norm(gf) * np.linalg.norm(rf) + 1e-30))
    print(f"[CONTRIB ffn (resid_post - resid_mid)] cos={cos_c:.6f}  "
          f"|kernel|={np.linalg.norm(gf):.4f}  |ref|={np.linalg.norm(rf):.4f}  "
          f"-> {'PASS' if cos_c >= 0.999 else 'FAIL'}")
    np.save("csl_ffn_output.npy", Z_layer0)

if __name__ == "__main__":
    main()
