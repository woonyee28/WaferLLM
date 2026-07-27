"""launch_ffn_verify.py — FFN stage of the disaggregated Decode layer (real Llama-3-8B block 0).

The decode block is validated as TWO device launches at different mesh sizes (see decode.csl
`ffn_only`): the ATTENTION stage (launch_verify.py, P=64) produces resid_mid; THIS stage (P=128,
ffn_only=1) takes resid_mid and produces resid_post = resid_mid + mlp(norm_post(resid_mid)).

The real 14336-wide up/gate/down weights do not fit per-PE at P=64 (~86 KB); at P=128 they fit
(~21 KB) with head_dim/P=1. The FFN is token-independent, so a single decode token's FFN output is
exactly one row of the full-sequence FFN — we validate the one decode-position row.

Input (resid_mid):
  --input <npy>  : a [dim] vector = the DEVICE Z_mid saved by launch_verify.py (true end-to-end,
                   used by run_layer_verify.sh). Default: the transformer_lens oracle resid_mid[pos]
                   (validates the FFN in isolation).

Run (after compiling the FFN config, e.g. via run_ffn_verify.sh):
    cs_python launch_ffn_verify.py --config model_config/llama8B_block0_ffn_p128.json
    cs_python launch_ffn_verify.py --config model_config/llama8B_block0_ffn_p128.json \
              --input csl_decode_zmid.npy --pos 64
"""
import os
import json
import argparse
import numpy as np

# Appliance/cluster device flow (mirrors Prefill): cloud artifact + client SdkRuntime(simulator=...).
from cerebras.sdk.client import SdkRuntime
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder

# Reuse the validated decode helpers (main-guarded; safe to import).
from launch_sim import sep
from launch_verify import load_block0_weights, report, report_contrib, cast_tensor_u32, d2h

OUT_PATH = "compile_out"  # cloud artifacts written by `python compile.py --mode device`


def parse_args():
    ap = argparse.ArgumentParser(description="Validate the Decode FFN stage vs transformer_lens (real Llama-3-8B block 0)")
    ap.add_argument("--config", default="model_config/llama8B_block0_ffn_p128.json")
    ap.add_argument("--input", default=None,
                    help="npy resid_mid ([dim] device Z_mid) to feed the FFN; default = oracle resid_mid[pos]")
    ap.add_argument("--pos", type=int, default=64,
                    help="oracle decode position to validate against (= attention stage prefill_len)")
    ap.add_argument("--oracle-dir", default=None, help="dir with resid_*_block0.npy (default: <repo>/pytorch/oracle_decode)")
    ap.add_argument("--simulator", action="store_true",
                    help="run the cloud artifact in the appliance SIMULATOR (default: real WSE-3)")
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    P          = cfg["P"]
    bsz        = cfg["bsz"]
    dim        = cfg["dim"]
    n_heads    = cfg["n_heads"]
    n_kv_heads = cfg["n_kv_heads"]
    head_dim   = cfg["head_dim"]
    ffn_dim    = cfg["ffn_dim"]
    assert cfg.get("ffn_only", 0) == 1, "launch_ffn_verify expects an ffn_only=1 config"
    assert bsz == 1, "verify harness assumes bsz=1 (one decode token)"

    dim_p_pe     = dim // P
    ffn_dim_p_pe = ffn_dim // P

    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    # ── Oracle ──────────────────────────────────────────────────────────────────
    if args.oracle_dir:
        oracle_dir = args.oracle_dir
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        oracle_dir = os.path.join(repo_root, "pytorch", "oracle_decode")
    resid_mid  = np.load(os.path.join(oracle_dir, "resid_mid_block0.npy")).astype(np.float32)
    resid_post = np.load(os.path.join(oracle_dir, "resid_post_block0.npy")).astype(np.float32)
    pos = args.pos
    o_mid  = resid_mid[pos]
    o_post = resid_post[pos]

    # ── FFN input (resid_mid) ─────────────────────────────────────────────────────
    if args.input is not None:
        x_in = np.load(args.input).astype(np.float16).ravel()
        if x_in.shape[0] != dim:               # allow a [seq, dim] array too
            x_in = x_in.reshape(-1, dim)[pos]
        print(f"[chain] FFN input = device Z_mid from {args.input}  (end-to-end)")
    else:
        x_in = o_mid.astype(np.float16)
        print(f"[standalone] FFN input = oracle resid_mid[{pos}]  (FFN validated in isolation)")

    print(f"Host: P={P} dim={dim} ffn_dim={ffn_dim} dim_p_pe={dim_p_pe} ffn_dim_p_pe={ffn_dim_p_pe} pos={pos}")

    # ── Real weights (only norm_post + up/gate/down are used by the FFN stage) ─────
    w = load_block0_weights(head_dim, n_heads, n_kv_heads)

    def tile_weight_row(W, rows, cols):
        return W.reshape(P, rows, P, cols).transpose(0, 2, 1, 3).reshape(P, P, rows * cols)

    UP_tile = tile_weight_row(w["up"],   dim_p_pe, ffn_dim_p_pe)
    GT_tile = tile_weight_row(w["gate"], dim_p_pe, ffn_dim_p_pe)
    DN_tile = tile_weight_row(w["down"], ffn_dim_p_pe, dim_p_pe)

    # Norm weight + X: dim-block by py, replicated across px (the decode residual-stream layout).
    tensor_W2 = np.tile(w["norm_post"].reshape(P, dim_p_pe), reps=(1, P))
    tensor_X  = np.tile(x_in.reshape(P, dim_p_pe),           reps=(1, P))

    # ── Runner (client API: cloud artifact; simulator=False -> real WSE-3, like Prefill) ──
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    with open(os.path.join(OUT_PATH, f"artifact_{cfg_name}.json"), encoding="utf-8") as f:
        artifact_id = json.load(f)["artifact_id"]

    with SdkRuntime(artifact_id, simulator=args.simulator) as runner:
        def h2d(name, flat_arr, count_per_pe):
            runner.memcpy_h2d(runner.get_id(name), cast_tensor_u32(flat_arr.ravel()), 0, 0, P, P, count_per_pe,
                              streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

        # Only the FFN inputs; attention weights/cache stay zero (unused when ffn_only=1).
        h2d("X",           tensor_X,  bsz * dim_p_pe)
        h2d("W2",          tensor_W2, dim_p_pe)
        h2d("UP_weight",   UP_tile,   dim_p_pe * ffn_dim_p_pe)
        h2d("GATE_weight", GT_tile,   dim_p_pe * ffn_dim_p_pe)
        h2d("DOWN_weight", DN_tile,   ffn_dim_p_pe * dim_p_pe)

        runner.launch("init_task", nonblock=False)
        runner.launch("decode_host", np.int16(1), np.int16(0), nonblock=False)   # 1 step

        post_grid = d2h(runner, runner.get_id("resid_post"), P, bsz, dim_p_pe, io_dtype, memcpy_order)

    def reconstruct(grid):
        out = np.zeros(P * dim_p_pe, dtype=np.float32)
        for py in range(P):
            out[py * dim_p_pe:(py + 1) * dim_p_pe] = grid[py, 0, :].astype(np.float32)
        return out

    kernel_post = reconstruct(post_grid)
    base = x_in.astype(np.float32)   # the resid_mid the FFN actually saw (its residual passthrough)

    sep(f"transformer_lens validation — FFN stage (P={P}), decode position {pos}")
    all_ok = True
    all_ok &= report("resid_post", kernel_post, o_post)
    # Contribution strips the residual passthrough: g = mlp(device input), r = oracle FFN delta.
    all_ok &= report_contrib("  ffn contribution", kernel_post, base + (o_post - o_mid), base)
    print()
    np.save("csl_decode_zpost.npy", kernel_post.astype(np.float16))
    print(f"  device Z_post saved -> csl_decode_zpost.npy")
    print(f"  Overall: {'ALL PASS ✓' if all_ok else 'SOME FAILURES — check above'}")


if __name__ == "__main__":
    main()
