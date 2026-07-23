#!/usr/bin/env python3
"""sweep_bss.py -- how does per-PE .bss scale with sequence length?

Recompiles each build at several seq_len values with P held FIXED, then reports the
worst-case per-PE .text / .data / .bss from the compiled ELFs. Runs BOTH the
attention (P=128, GQA) and FFN (P=256) builds in one pass.

Only the activation buffers depend on seq_len (via seq_len_p_pe = seq_len / P):
  * linear  in seq_len_p_pe : X_tile, h1/h2, z1/z2/z3, Z*, seqLen_* tmps, XQ/XK/XV ...
  * QUADRATIC in seq_len_p_pe: score, seqLen_seqLen_tmp   (the seq x seq attention tile,
                               attention build only)
Code (.text) and the weight tiles (Q/K/V/O, up/gate/down) are seq_len-independent,
so any growth you see in .bss is pure activation scaling.

Run on the SDK host (needs cslc + size on PATH):
    python sweep_bss.py

Leaves out_<cfg>/ dirs and model_config/_sweep_*.json behind for inspection.
"""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, "model_config")

# Each build keeps P fixed and sweeps seq_len = P * seq_len_p_pe.
#   attention: real Llama-3-8B GQA; ffn_dim truncated to 512 so the FFN placeholder
#              fits (exactly like llama8B_block0_p128_probe).
#   ffn      : real Llama-3-8B FFN (dim=4096, ffn_dim=14336); attention params are
#              P=256-divisible placeholders (unused when ffn_only=1).
BUILDS = [
    ("attention  P=128  GQA",
     dict(P=128, dim=4096, n_heads=32, n_kv_heads=8, head_dim=128, ffn_dim=512, ffn_only=0)),
    ("ffn        P=256",
     dict(P=256, dim=4096, n_heads=16, n_kv_heads=4, head_dim=256, ffn_dim=14336, ffn_only=1)),
]
SPP = [1, 2, 4, 8, 16]   # seq_len_p_pe values to sweep (seq_len = P * spp)


def worst_pe(out_dir):
    """Largest-footprint compute PE: (text, data, bss, total) in bytes."""
    best = None
    for e in glob.glob(os.path.join(out_dir, "bin", "*.elf")):
        cols = subprocess.check_output(["size", e]).decode().splitlines()[1].split()
        text, data, bss, dec = (int(cols[0]), int(cols[1]), int(cols[2]), int(cols[3]))
        if best is None or dec > best[3]:
            best = (text, data, bss, dec)
    return best


def sweep(tag, base):
    P = base["P"]
    hdr = f"{'seq_len':>8} {'seq/P':>6} {'.text':>8} {'.data':>7} {'.bss':>8} {'total':>8} {'d.bss':>7}"
    print(f"\n=== {tag} ===")
    print(hdr)
    print("-" * len(hdr))
    prev_bss = None
    slug = tag.split()[0]                       # "attention" / "ffn"
    for spp in SPP:
        seq = P * spp
        cfg_name = f"_sweep_{slug}_s{seq}"
        cfg_path = os.path.join(CFG_DIR, cfg_name + ".json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(dict(base, seq_len=seq), f)

        r = subprocess.run(
            [sys.executable, "compile.py", "--mode", "sim", "--config", cfg_path],
            cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if r.returncode != 0:
            print(f"{seq:>8} {spp:>6}   COMPILE FAILED (SRAM overflow? this is the ceiling)")
            break

        text, data, bss, dec = worst_pe(os.path.join(HERE, f"out_{cfg_name}"))
        dbss = "" if prev_bss is None else f"+{bss - prev_bss}"
        print(f"{seq:>8} {spp:>6} {text:>8} {data:>7} {bss:>8} {dec:>8} {dbss:>7}")
        prev_bss = bss


def main():
    for tag, base in BUILDS:
        sweep(tag, base)


if __name__ == "__main__":
    main()
