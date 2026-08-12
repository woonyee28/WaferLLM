import numpy as np
import argparse
import glob
import struct
import os
import json

from cerebras.sdk.sdk_utils import input_array_to_u32
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder

# stages dispatched by prefill_struct(), in `flag` order (0..15 plus the else tail).
# 17 stages -> 18 boundary timestamps -> 17 deltas.
stages = [
    "rmsnorm_x", "xq_matmul", "xk_matmul", "xv_matmul", "xq_rope", "xk_rope",
    "score_matmul", "softmax_score", "output_matmul", "h1_matmul", "z_add",
    "rmsnorm_z", "z1_matmul", "z2_matmul", "z3_comp", "h2_matmul", "add_result",
]

n_bounds = 18
stage_f32 = n_bounds * 3 // 2  # 3 u16 per timestamp, packed 2 u16 per f32

def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])

def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)

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
        
# Tiles worth capturing as a numerical baseline. XQ/XK are the rope outputs and
# are the tightest check for a rope overlay; Z is the end-to-end result.
BASELINE_TILES = ("X_tile", "X_norm_tile", "XQ_tile", "XK_tile", "XV_tile",
                  "output_tile", "h1_tile", "h2_tile", "Z_tile")

# Symbols that are not (seq_len_p_pe x dim_p_pe) tiles: captured raw, per PE,
# with no reassembly. Each entry is (csl_name, numpy dtype).
FLAT_SYMBOLS = (("slot_buf", np.uint32),)

# Matches --fabric-offsets=4,1 in compile.py: the P x P core rectangle starts here.
CORE_OFFSET = (4, 1)

# Baselines live outside out_<cfg>, which compile.py deletes on every build.
BASELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline")


def _resolve_symbols(bin_dir, wanted):
    """Map CSL names to their ELF names.

    The compiler renames module-level arrays to '$$csl_base_address$$<n>$$<name>',
    and debug_util looks symbols up literally, so the plain name never resolves.
    """
    from cerebras.elf.cself import ELFLoader
    resolved = {}
    for elf in sorted(glob.glob(os.path.join(bin_dir, "out_*.elf"))):
        try:
            syms = ELFLoader(elf_file=elf).symbols
        except Exception:
            continue
        names = list(syms.keys()) if hasattr(syms, "keys") else list(syms)
        for w in wanted:
            if w in resolved:
                continue
            for n in names:
                if n == w or n.endswith("$$" + w):
                    resolved[w] = n
                    break
        if len(resolved) == len(wanted):
            break
    return resolved


def dump_tiles(P, seq_len_p_pe, dim_p_pe, tag):
    """Save the on-PE tiles as numpy, for use as an overlay verification baseline.

    Must run in-process after runner.stop(): debug_util reads the live simulator's
    memory, so a separate post-hoc process has nothing to read. Nothing here is
    exported from prefill.csl, so the binary is unaffected by this capture.

    A tile is stored transposed on the PE, so both the raw (P, P, N) tiles and the
    reassembled (seq_len, dim) matrices are saved. Prefer the raw tiles when
    comparing runs: they carry no layout assumption.
    """
    from cerebras.sdk.debug.debug_util import debug_util

    out_dir = os.getcwd()
    resolved = _resolve_symbols(os.path.join(out_dir, "bin"), BASELINE_TILES)
    debug = debug_util(out_dir)

    raw, assembled = {}, {}
    for sym in BASELINE_TILES:
        elf_name = resolved.get(sym)
        if elf_name is None:
            print(f"  {sym:<14} SKIP (no such symbol; inlined away?)")
            continue
        tiles = np.asarray(debug.get_symbol_rect(
            (CORE_OFFSET, (P, P)), elf_name, np.float16))
        raw[sym] = tiles

        mat = np.zeros((P * seq_len_p_pe, P * dim_p_pe), dtype=np.float16)
        for x in range(P):
            for y in range(P):
                mat[y * seq_len_p_pe:(y + 1) * seq_len_p_pe,
                    x * dim_p_pe:(x + 1) * dim_p_pe] = \
                    tiles[x, y].reshape(dim_p_pe, seq_len_p_pe).T
        assembled[sym] = mat
        print(f"  {sym:<14} {tiles.shape} -> {mat.shape}  "
              f"min={np.nanmin(mat):.4g} max={np.nanmax(mat):.4g}")

    flat_resolved = _resolve_symbols(os.path.join(out_dir, "bin"),
                                     [s for s, _ in FLAT_SYMBOLS])
    for sym, dtype in FLAT_SYMBOLS:
        elf_name = flat_resolved.get(sym)
        if elf_name is None:
            continue
        vals = np.asarray(debug.get_symbol_rect(
            (CORE_OFFSET, (P, P)), elf_name, dtype))
        raw[sym] = vals
        print(f"  {sym:<14} {vals.shape} {vals.dtype}  "
              f"first PE head={[hex(int(v)) for v in vals[0, 0][:3]]}")

    raw_path = os.path.join(BASELINE_DIR, f"{tag}_tiles_raw.npz")
    mat_path = os.path.join(BASELINE_DIR, f"{tag}_tiles.npz")
    np.savez(raw_path, **raw)
    np.savez(mat_path, **assembled)
    print(f"  wrote {raw_path}\n  wrote {mat_path}")

    # X_tile is never written by the pass, so it must still equal the host input.
    # That round-trip is what validates the reassembly above.
    if "X_tile" in assembled:
        ref = np.load(os.path.join(BASELINE_DIR, f"{tag}_inputs.npz"))["X"]
        ok = np.array_equal(assembled["X_tile"], ref)
        print(f"  layout round-trip on X_tile: {'PASS' if ok else 'FAIL'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prefill on simulator")
    parser.add_argument("--config", default="config.json", type=str, help="Config file")
    parser.add_argument("--scale", default=1.0, type=float,
                        help="Scale factor on the random weights. The default 1.0 "
                             "overflows fp16 by the end of the FFN, leaving Z_tile "
                             "all-inf; use a smaller value for a usable end-to-end "
                             "numerical baseline.")
    parser.add_argument("--seed", default=0, type=int,
                        help="RNG seed for the random inputs, so a run is reproducible "
                             "and can serve as a numerical baseline")
    args = parser.parse_args()
    return args

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
    
    np.random.seed(args.seed)
    s = args.scale

    tensor_X = np.random.rand(seq_len, dim).astype(np.float16)

    W = np.random.rand(1, dim).astype(np.float16)
    tensor_W = np.tile(W.reshape(P, dim_p_pe), reps=(1, P))

    tensor_q_weight = (s * np.random.rand(dim, dim)).astype(np.float16)
    tensor_k_weight = (s * np.random.rand(dim, dim)).astype(np.float16)
    tensor_v_weight = (s * np.random.rand(dim, dim)).astype(np.float16)

    freqs_sin = np.random.rand(1, P*_dim_p_pe//2).astype(np.float16)
    tensor_freqs_sin = np.tile(freqs_sin.reshape(P, _dim_p_pe//2), reps=(1, P))
    freqs_cos = np.random.rand(1, P*_dim_p_pe//2).astype(np.float16)
    tensor_freqs_cos = np.tile(freqs_cos.reshape(P, _dim_p_pe//2), reps=(1, P))

    tensor_o_weight = (s * np.random.rand(dim, dim)).astype(np.float16)
    tensor_up_weight = (s * np.random.rand(dim, ffn_dim)).astype(np.float16)
    tensor_gate_weight = (s * np.random.rand(dim, ffn_dim)).astype(np.float16)
    tensor_down_weight = (s * np.random.rand(ffn_dim, dim)).astype(np.float16)
    
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

    # Persist the exact inputs so any later run (overlay or not) can be checked
    # against this one without re-deriving them.
    tag = f"{cfg_name}_seed{args.seed}_scale{args.scale:g}"
    os.makedirs(BASELINE_DIR, exist_ok=True)
    np.savez(os.path.join(BASELINE_DIR, f"{tag}_inputs.npz"),
             seed=np.int64(args.seed), scale=np.float64(args.scale),
             X=tensor_X, W=tensor_W,
             q_weight=tensor_q_weight, k_weight=tensor_k_weight,
             v_weight=tensor_v_weight, o_weight=tensor_o_weight,
             up_weight=tensor_up_weight, gate_weight=tensor_gate_weight,
             down_weight=tensor_down_weight,
             freqs_sin=tensor_freqs_sin, freqs_cos=tensor_freqs_cos)

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
    symbol_stage_time = runner.get_id("stage_time")
    
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

    # Per-stage boundary timestamps. stage_tsc is overwritten every pass, so
    # what survives is the final (fully warmed) pass.
    stage_1d_f32 = np.zeros(P*P*stage_f32, dtype=np.float32)
    runner.memcpy_d2h(stage_1d_f32, symbol_stage_time, 0, 0, P, P, stage_f32, streaming=False,
                    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

    runner.stop()

    dump_tiles(P, seq_len_p_pe, dim_p_pe, tag)

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

    # Per-stage cycle breakdown (single warmed pass, per PE)
    u32 = stage_1d_f32.view(np.uint32).reshape(P, P, stage_f32)
    words = np.empty((P, P, stage_f32 * 2), dtype=np.uint16)
    words[..., 0::2] = (u32 & 0xFFFF).astype(np.uint16)
    words[..., 1::2] = (u32 >> 16).astype(np.uint16)

    w3 = words.reshape(P, P, n_bounds, 3).astype(np.int64)
    stage_ts = w3[..., 0] + (w3[..., 1] << 16) + (w3[..., 2] << 32)

    # Intra-PE deltas: launch skew cancels, so time_ref must NOT be subtracted.
    stage_cycles = np.diff(stage_ts, axis=-1)

    py, px = P // 2, P // 2
    pe_cycles = stage_cycles[py, px]
    pe_total = int(pe_cycles.sum())

    print(f"\nPer-stage cycles (PE {px},{py}, final pass)")
    print(f"{'stage':<16}{'cycles':>12}{'%':>8}{'grid min':>12}{'grid max':>12}")
    for i, name in enumerate(stages):
        c = int(pe_cycles[i])
        pct = 100.0 * c / pe_total if pe_total else 0.0
        print(f"{name:<16}{c:>12}{pct:>7.1f}%"
              f"{int(stage_cycles[..., i].min()):>12}{int(stage_cycles[..., i].max()):>12}")
    print(f"{'TOTAL':<16}{pe_total:>12}")

if __name__ == "__main__":
    main()