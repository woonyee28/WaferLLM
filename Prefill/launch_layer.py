#!/usr/bin/env python3
"""
launch_layer.py -- one disaggregated, end-to-end Llama-3-8B block-0 layer on WSE-3.

The full layer does not fit in PE SRAM at any single P, so it runs as TWO device
launches with DIFFERENT mesh sizes, chained through the device's own activations:

    Stage 1  ATTENTION  P=128  (ffn_only=0)   resid_pre --> Z_mid   (= resid_mid)
    Stage 2  FFN        P=256  (ffn_only=1)   Z_mid     --> Z_final (= resid_post)

Why the two stages use different P (and why that is correct, not a bug):
  * Attention head_dim=128 needs head_dim_p_pe = 128/P >= 1  ->  P <= 128.
  * FFN's 14336-wide up/gate/down weights only fit each PE's 48 KB SRAM at P >= 256
    (OOM at P=128; that is why ffn_p128_probe uses a TRUNCATED ffn_dim=512).
    These two constraints are incompatible -> the layer must be disaggregated.
  * seq_len differs (128 vs 256) but that is only zero-padding up to the mesh height:
    the prompt is 7 real tokens, and the FFN is token-INDEPENDENT (its matmuls contract
    dim / ffn_dim, never seq_len -- see prefill.csl z1/z2/h2 vs attention's output_matmul),
    so each token's FFN output is identical regardless of how many padded rows share the
    mesh. We therefore thread only the real tokens from Stage 1's Z_mid into Stage 2.

This driver runs both stages (reusing launch_device.py and launch_ffn.py; the only source
change is launch_ffn's --input flag), feeds the DEVICE Z_mid into the FFN, then reports
Z_mid and Z_final against the transformer_lens checkpoints -- cosine similarity AND actual
values -- for the single layer.

Prerequisites (both artifacts must already be compiled):
    compile_out/artifact_<attn-config>.json   (e.g. llama8B_block0_p128_probe)
    compile_out/artifact_<ffn-config>.json    (e.g. ffn_p256)

Usage:
    python launch_layer.py                                  # default configs
    python launch_layer.py --attn-config model_config/llama8B_block0_p128_probe.json \
                           --ffn-config  model_config/ffn_p256.json
    python launch_layer.py --skip-run                       # re-report from saved npy only
"""
import os
import sys
import argparse
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESID_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "pytorch")

ATTN_ZMID_NPY = "csl_layer0_resid_mid.npy"    # saved by launch_device.py (device Z_mid, padded)
ATTN_ZMID_REAL_NPY = "csl_attn_zmid_real.npy" # sliced to the real tokens, handed to the FFN
FFN_OUT_NPY = "csl_ffn_output.npy"            # saved by launch_ffn.py (device Z_final)


def cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def report(name, got, ref, base_got=None, base_ref=None):
    """Report device vs transformer_lens: full + contribution cosine, and actual values."""
    got = got.astype(np.float64)
    ref = ref.astype(np.float64)
    n = ref.shape[0]
    got = got[:n]

    full = cos(got, ref)
    print(f"\n[{name}]  tokens={n}  dim={ref.shape[1]}")
    print(f"  full cosine        : {full:.6f}   -> {'PASS' if full >= 0.999 else 'FAIL'}")

    if base_got is not None:
        # contribution cosine strips the residual passthrough (the honest metric): a small
        # correction rides on a large passthrough, so the full cosine flatters a wrong block.
        if base_ref is None:
            base_ref = base_got
        c = cos(got - base_got.astype(np.float64)[:n], ref - base_ref.astype(np.float64)[:n])
        print(f"  contribution cosine: {c:.6f}   -> {'PASS' if c >= 0.999 else 'FAIL'}   "
              f"(strips residual passthrough)")

    err = np.abs(got - ref)
    print(f"  |device|={np.linalg.norm(got):.4f}  |oracle|={np.linalg.norm(ref):.4f}  "
          f"max_abs={err.max():.3e}  mean_abs={err.mean():.3e}")

    # actual values: first token's first 8 dims, plus its largest-magnitude dim
    j = int(np.argmax(np.abs(ref[0])))
    with np.printoptions(precision=4, suppress=True):
        print(f"  token0 dims[0:8] device: {got[0, :8]}")
        print(f"  token0 dims[0:8] oracle: {ref[0, :8]}")
    print(f"  token0 peak dim {j}: device={got[0, j]:+.4f}  oracle={ref[0, j]:+.4f}")


def run_stage(cmd):
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"[launch_layer] stage FAILED (exit {r.returncode}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description="Disaggregated full-layer run + validation")
    ap.add_argument("--attn-config", default="model_config/llama8B_block0_p128_probe.json",
                    help="attention artifact config (P=128, ffn_only=0)")
    ap.add_argument("--ffn-config", default="model_config/ffn_p256.json",
                    help="FFN artifact config (P=256, ffn_only=1)")
    ap.add_argument("--skip-run", action="store_true",
                    help="skip both device launches; just re-report from existing npy outputs")
    args = ap.parse_args()

    oracle_pre = np.load(os.path.join(RESID_DIR, "resid_pre_block0.npy")).astype(np.float32)
    oracle_mid = np.load(os.path.join(RESID_DIR, "resid_mid_block0.npy")).astype(np.float32)
    oracle_post = np.load(os.path.join(RESID_DIR, "resid_post_block0.npy")).astype(np.float32)
    n_tok = oracle_mid.shape[0]

    if not args.skip_run:
        # ---- Stage 1: attention (P=128) -> device Z_mid (saved to ATTN_ZMID_NPY) ----
        run_stage([sys.executable, "launch_device.py", "--config", args.attn_config])

        # thread only the real tokens of the DEVICE Z_mid into the FFN (see module docstring)
        zmid_dev_padded = np.load(os.path.join(HERE, ATTN_ZMID_NPY)).astype(np.float16)
        np.save(os.path.join(HERE, ATTN_ZMID_REAL_NPY), zmid_dev_padded[:n_tok])

        # ---- Stage 2: FFN (P=256) fed the DEVICE Z_mid -> device Z_final ----
        run_stage([sys.executable, "launch_ffn.py",
                   "--config", args.ffn_config, "--input", ATTN_ZMID_REAL_NPY])

    # ---- combined validation vs transformer_lens (single layer) ----
    zmid_dev = np.load(os.path.join(HERE, ATTN_ZMID_NPY)).astype(np.float32)[:n_tok]
    zfinal_dev = np.load(os.path.join(HERE, FFN_OUT_NPY)).astype(np.float32)[:n_tok]

    print("\n" + "=" * 74)
    print("  DISAGGREGATED FULL LAYER   attention P=128  ->  FFN P=256   vs transformer_lens")
    print("=" * 74)
    # Z_mid contribution rides on resid_pre; Z_final's FFN contribution rides on the input it
    # actually saw (device Z_mid) vs the oracle FFN contribution (resid_post - resid_mid).
    report("Z_mid   (resid_mid, after attention)", zmid_dev, oracle_mid, base_got=oracle_pre)
    report("Z_final (resid_post, after FFN)", zfinal_dev, oracle_post,
           base_got=zmid_dev, base_ref=oracle_mid)
    print()


if __name__ == "__main__":
    main()
