import numpy as np
import argparse
import struct
import os
import json

from cerebras.sdk.sdk_utils import input_array_to_u32
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder

def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])

def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)

# wyn: add function to untile the processed Z
def untile_flat_1d(input_flat_1d, P, seq_len_p_pe, dim_p_pe):
    # On each PE, it stores (seq_len_p_pe,dim_p_pe)
    # d2h copy will return flat array in row major.
    a = input_flat_1d.reshape(P, P, dim_p_pe, seq_len_p_pe)  # py=seq-block, px=dim-block
    a = a.transpose(0, 3, 1, 2)  # (py, s_local, px, d_local)
    return a.reshape(seq_len_p_pe * P, dim_p_pe * P) # (seq_len, dim)
# wyn: end

def assignId(pc, P):
    send_id = 0
    recv_id = 0
    
    pc = pc + 1
    
    if pc%2 == 0:
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
        
    if P%2 == 0:
        if pc == P-1:
            send_id = P
            recv_id = P - 3
        if pc == P:
            send_id = P - 2
            recv_id = P - 1
    else:
        if pc == P-1:
            send_id = max(send_id, 1)
            recv_id = P
        if pc == P:
            send_id = P - 1
            recv_id = P - 2
    return send_id - 1, recv_id - 1

class Config:
    def __init__(self):
        self.P = 8
        self.dim = 64
        self.n_heads = 1
        self.n_kv_heads = 1
        self.head_dim = 64
        self.seq_len = 64
        self.ffn_dim = 64
        
def parse_args():
    parser = argparse.ArgumentParser(description="Prefill on simulator")
    parser.add_argument("--config", default="config.json", type=str, help="Config file")
    args = parser.parse_args()
    return args

# wyn: load real Meta-Llama-3-8B block-0 weights from the safetensors checkpoint.
# HF Linear stores weights as [out, in] and applies x @ W.T; WaferLLM does x @ W,
# so the 2-D projection weights are transposed. The 1-D norm weights are left as-is.
def load_block0_weights():
    import glob, torch
    from safetensors import safe_open

    model_dir = "/home/woonyee/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B"
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

    return {
        "q"        : get("self_attn.q_proj.weight"),
        "k"        : get("self_attn.k_proj.weight"),
        "v"        : get("self_attn.v_proj.weight"),
        "o"        : get("self_attn.o_proj.weight"),
        "up"       : get("mlp.up_proj.weight"),
        "gate"     : get("mlp.gate_proj.weight"),
        "down"     : get("mlp.down_proj.weight"),
        "norm_pre" : get("input_layernorm.weight", transpose=False),
        "norm_post": get("post_attention_layernorm.weight", transpose=False),
    }
# wyn: end

def main():
    args = parse_args()
    config = Config()
    
    if not os.path.exists(args.config):
        print("Host: Use default test values.")
    else:
        with open(args.config) as f:
            config.__dict__.update(json.load(f))
            
    P = config.P
    dim = config.dim
    seq_len = config.seq_len
    ffn_dim = config.ffn_dim
    
    dim_p_pe = dim // P
    seq_len_p_pe = seq_len // P
    ffn_dim_p_pe = ffn_dim // P
    
    _dim_p_pe = dim_p_pe
    if (dim_p_pe % 2) == 1:
        _dim_p_pe = dim_p_pe - 1
        
    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR
    
    # wyn: real Llama-3-8B block-0 weights + input X (baseline.py dump) instead of random.
    weights = load_block0_weights()

    # input X = block-0 resid_pre, zero-padded up to seq_len (which must be a multiple of P)
    resid_dir = "/home/woonyee/Cerebras/pytorch"
    resid_pre = np.load(os.path.join(resid_dir, "resid_pre_block0.npy")).astype(np.float16)  # [n_tok, dim]
    # check the current prompt length
    n_tok = resid_pre.shape[0]
    # if too long then increase the seq_len in the config.json
    assert n_tok <= seq_len, f"prompt has {n_tok} tokens > seq_len {seq_len}; raise seq_len in the config"
    # allocate the full size input the kernel expects: [128,4096], all zeros at first.
    tensor_X = np.zeros((seq_len, dim), dtype=np.float16)
    tensor_X[:n_tok] = resid_pre

    W = weights["norm_pre"].reshape(1, dim)
    tensor_W = np.tile(W.reshape(P, dim_p_pe), reps=(1, P))

    tensor_q_weight = weights["q"]
    tensor_k_weight = weights["k"]
    tensor_v_weight = weights["v"]
    # wyn: end

    freqs_sin = np.random.rand(1, P*_dim_p_pe//2).astype(np.float16)
    tensor_freqs_sin = np.tile(freqs_sin.reshape(P, _dim_p_pe//2), reps=(1, P))
    freqs_cos = np.random.rand(1, P*_dim_p_pe//2).astype(np.float16)
    tensor_freqs_cos = np.tile(freqs_cos.reshape(P, _dim_p_pe//2), reps=(1, P))

    # wyn: real block-0 weights (loaded above)
    tensor_o_weight = weights["o"]
    tensor_up_weight = weights["up"]
    tensor_gate_weight = weights["gate"]
    tensor_down_weight = weights["down"]
    # wyn: end
    
    ind = np.zeros((P, P)).astype(int)
    
    for i in range(P):
        for j in range(P):
            if i == 0:
                ind[0, j] = j
            elif i == 1:
                _, ind[1, j] = assignId(ind[0, j], P)
            else:
                if (i-1)%2==0:
                    _, ind[i, j] = assignId(ind[i-2, j], P)
                else:
                    ind[i, j], _ = assignId(ind[i-2, j], P)
                    
    tensor_q_weight_shifted = np.zeros((dim, dim)).astype(np.float16)
    tensor_k_weight_shifted = np.zeros((dim, dim)).astype(np.float16)
    tensor_v_weight_shifted = np.zeros((dim, dim)).astype(np.float16)
    
    tensor_o_weight_shifted = np.zeros((dim, dim)).astype(np.float16)
    tensor_up_weight_shifted = np.zeros((dim, ffn_dim)).astype(np.float16)
    tensor_gate_weight_shifted = np.zeros((dim, ffn_dim)).astype(np.float16)
    tensor_down_weight_shifted = np.zeros((ffn_dim, dim)).astype(np.float16)
    
    for i in range(P):
        for j in range(P):
            t = ind[i, j]
            tensor_q_weight_shifted[i*dim_p_pe:(i+1)*dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe] = tensor_q_weight[t*dim_p_pe:(t+1)*dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe]
            tensor_k_weight_shifted[i*dim_p_pe:(i+1)*dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe] = tensor_k_weight[t*dim_p_pe:(t+1)*dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe]
            tensor_v_weight_shifted[i*dim_p_pe:(i+1)*dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe] = tensor_v_weight[t*dim_p_pe:(t+1)*dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe]
            
            tensor_o_weight_shifted[i*dim_p_pe:(i+1)*dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe] = tensor_o_weight[t*dim_p_pe:(t+1)*dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe]
            tensor_up_weight_shifted[i*dim_p_pe:(i+1)*dim_p_pe, j*ffn_dim_p_pe:(j+1)*ffn_dim_p_pe] = tensor_up_weight[t*dim_p_pe:(t+1)*dim_p_pe, j*ffn_dim_p_pe:(j+1)*ffn_dim_p_pe]
            tensor_gate_weight_shifted[i*dim_p_pe:(i+1)*dim_p_pe, j*ffn_dim_p_pe:(j+1)*ffn_dim_p_pe] = tensor_gate_weight[t*dim_p_pe:(t+1)*dim_p_pe, j*ffn_dim_p_pe:(j+1)*ffn_dim_p_pe]
            tensor_down_weight_shifted[i*ffn_dim_p_pe:(i+1)*ffn_dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe] = tensor_down_weight[t*ffn_dim_p_pe:(t+1)*ffn_dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe]
            
    
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    # Run from inside out_<cfg> so simfab run artifacts (sim.log, sim_stats.json,
    # simconfig.json, simfab_traces/, out.core, wio_flows_tmpdir.*) land there
    # instead of polluting the module root. SdkRuntime has no output-dir kwarg;
    # it writes these to CWD. compile.py already populated out_<cfg> via `cslc -o`.
    out_dir = os.path.abspath(f"out_{cfg_name}")
    if not os.path.isdir(out_dir):
        raise SystemExit(f"Host: {out_dir} not found — run compile.py --mode sim first")
    os.chdir(out_dir)
    runner = SdkRuntime(out_dir)
    runner.load()
    runner.run()
    
    sym_X = runner.get_id("X")
    sym_W = runner.get_id("W")
    sym_Q_weight = runner.get_id("Q_weight")
    sym_K_weight = runner.get_id("K_weight")
    sym_V_weight = runner.get_id("V_weight")
    sym_freqs_sin = runner.get_id("freqs_sin")
    sym_freqs_cos = runner.get_id("freqs_cos")
    sym_O_weight = runner.get_id("O_weight")
    sym_UP_weight = runner.get_id("UP_weight")
    sym_GATE_weight = runner.get_id("GATE_weight")
    sym_DOWN_weight = runner.get_id("DOWN_weight")
    
    symbol_time_memcpy = runner.get_id("time_memcpy")
    symbol_time_ref = runner.get_id("time_ref")
    
    Xc1 = tensor_X.reshape(P, seq_len_p_pe, P, dim_p_pe)
    Xc2 = Xc1.transpose(0, 2, 3, 1)
    Xc3 = Xc2.reshape(P, P, seq_len_p_pe * dim_p_pe)
    Xc_u32 = input_array_to_u32(Xc3.ravel(), 1, 1)
    runner.memcpy_h2d(sym_X, Xc_u32, 0, 0, P, P, seq_len_p_pe * dim_p_pe, \
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)
    
    W_u32 = input_array_to_u32(tensor_W.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_W, W_u32, 0, 0, P, P, dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    Q_reshape = tensor_q_weight_shifted.reshape(P, dim_p_pe, P, dim_p_pe)
    Q_transpose = Q_reshape.transpose(0, 2, 1, 3)
    Q_reshape = Q_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
    Q_u32 = input_array_to_u32(Q_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_Q_weight, Q_u32, 0, 0, P, P, dim_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    K_reshape = tensor_k_weight_shifted.reshape(P, dim_p_pe, P, dim_p_pe)
    K_transpose = K_reshape.transpose(0, 2, 1, 3)
    K_reshape = K_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
    K_u32 = input_array_to_u32(K_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_K_weight, K_u32, 0, 0, P, P, dim_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    V_reshape = tensor_v_weight_shifted.reshape(P, dim_p_pe, P, dim_p_pe)
    V_transpose = V_reshape.transpose(0, 2, 1, 3)
    V_reshape = V_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
    V_u32 = input_array_to_u32(V_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_V_weight, V_u32, 0, 0, P, P, dim_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    freqs_sin_u32 = input_array_to_u32(tensor_freqs_sin.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_freqs_sin, freqs_sin_u32, 0, 0, P, P, _dim_p_pe//2, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )

    freqs_cos_u32 = input_array_to_u32(tensor_freqs_cos.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_freqs_cos, freqs_cos_u32, 0, 0, P, P, _dim_p_pe//2, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    O_reshape = tensor_o_weight_shifted.reshape(P, dim_p_pe, P, dim_p_pe)
    O_transpose = O_reshape.transpose(0, 2, 1, 3)
    O_reshape = O_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
    O_u32 = input_array_to_u32(O_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_O_weight, O_u32, 0, 0, P, P, dim_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    UP_reshape = tensor_up_weight_shifted.reshape(P, dim_p_pe, P, ffn_dim_p_pe)
    UP_transpose = UP_reshape.transpose(0, 2, 1, 3)
    UP_reshape = UP_transpose.reshape(P, P, dim_p_pe * ffn_dim_p_pe)
    UP_u32 = input_array_to_u32(UP_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_UP_weight, UP_u32, 0, 0, P, P, dim_p_pe * ffn_dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    GATE_reshape = tensor_gate_weight_shifted.reshape(P, dim_p_pe, P, ffn_dim_p_pe)
    GATE_transpose = GATE_reshape.transpose(0, 2, 1, 3)
    GATE_reshape = GATE_transpose.reshape(P, P, dim_p_pe * ffn_dim_p_pe)
    GATE_u32 = input_array_to_u32(GATE_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_GATE_weight, GATE_u32, 0, 0, P, P, dim_p_pe * ffn_dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    DOWN_reshape = tensor_down_weight_shifted.reshape(P, ffn_dim_p_pe, P, dim_p_pe)
    DOWN_transpose = DOWN_reshape.transpose(0, 2, 1, 3)
    DOWN_reshape = DOWN_transpose.reshape(P, P, ffn_dim_p_pe * dim_p_pe)
    DOWN_u32 = input_array_to_u32(DOWN_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_DOWN_weight, DOWN_u32, 0, 0, P, P, ffn_dim_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    runner.launch('init_task', nonblock=False)
    total_warmup_times, total_repeat_times = 1, 3
    runner.launch('prefill_host', np.int16(total_warmup_times), np.int16(total_repeat_times), nonblock=False)
    
    time_memcpy_1d_f32 = np.zeros(P*P*3, dtype=np.float32)
    runner.memcpy_d2h(time_memcpy_1d_f32, symbol_time_memcpy, 0, 0, P, P, 3, streaming=False,
                    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (P, P, 3), order='C')
    
    time_ref_1d_f32 = np.zeros(P*P*2, np.float32)
    runner.memcpy_d2h(time_ref_1d_f32, symbol_time_ref, 0, 0, P, P, 2, streaming=False,
                    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    time_ref_hwl = np.reshape(time_ref_1d_f32, (P, P, 2), order='C')

    # wyn: read back the layer output Z, after all processing for baseline verification.
    sym_Z = runner.get_id("Z")
    Z_1d = np.zeros(P * P * seq_len_p_pe * dim_p_pe, dtype=np.float16)
    runner.memcpy_d2h(Z_1d, sym_Z, 0, 0, P, P, seq_len_p_pe * dim_p_pe,
                      streaming=False, order=memcpy_order, data_type=io_dtype, nonblock=False)
    Z_layer0 = untile_flat_1d(Z_1d, P, seq_len_p_pe, dim_p_pe)   # (seq_len, dim)
    np.save("csl_layer0_output.npy", Z_layer0)

    # diff against transformer_lens block-0 resid_post (real token rows only; padded rows ignored)
    resid_post = np.load(os.path.join(resid_dir, "resid_post_block0.npy")).astype(np.float32)
    got = Z_layer0[:resid_post.shape[0]].astype(np.float32)
    abs_err = np.abs(got - resid_post)
    tol = 2e-2
    ok = np.all(abs_err <= tol + tol * np.abs(resid_post))
    print(f"[diff vs resid_post] max_abs={abs_err.max():.4e} mean_abs={abs_err.mean():.4e} -> {'PASS' if ok else 'FAIL'}")
    # wyn: end

    runner.stop()
    
    time_start = np.zeros((P, P)).astype(int)
    time_end = np.zeros((P, P)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(P):
        for h in range(P):
            hex_t0 = int(float_to_hex(time_memcpy_hwl[(h, w, 0)]), base=16)
            hex_t1 = int(float_to_hex(time_memcpy_hwl[(h, w, 1)]), base=16)
            hex_t2 = int(float_to_hex(time_memcpy_hwl[(h, w, 2)]), base=16)
            word[0] = hex_t0 & 0x0000ffff
            word[1] = (hex_t0 >> 16) & 0x0000ffff
            word[2] = hex_t1 & 0x0000ffff
            time_start[(h, w)] = make_u48(word)
            word[0] = (hex_t1 >> 16) & 0x0000ffff
            word[1] = hex_t2 & 0x0000ffff
            word[2] = (hex_t2 >> 16) & 0x0000ffff
            time_end[(h, w)] = make_u48(word)
    
    time_ref = np.zeros((P, P)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(P):
        for h in range(P):
            hex_t0 = int(float_to_hex(time_ref_hwl[(h, w, 0)]), base=16)
            hex_t1 = int(float_to_hex(time_ref_hwl[(h, w, 1)]), base=16)
            word[0] = hex_t0 & 0x0000ffff
            word[1] = (hex_t0 >> 16) & 0x0000ffff
            word[2] = hex_t1 & 0x0000ffff
            time_ref[(h, w)] = make_u48(word)
            
    for py in range(P):
        for px in range(P):
            time_ref[(py, px)] = time_ref[(py, px)] - (px + py)
            
    time_start = time_start - time_ref
    time_end = time_end - time_ref
    
    min_time_start = time_start.min()
    max_time_end = time_end.max()
    
    print(f"\nRepeat count: {total_repeat_times}")
    print(f"Mean cycle count: {np.mean(time_end - time_start)/total_repeat_times}")
    print(f"Max Cycle count: {(max_time_end - min_time_start)/total_repeat_times}")
    
if __name__ == "__main__":
    main()
