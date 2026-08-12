"""Unified compile entry for Prefill (config-driven).

  --mode sim     : local `cslc` build into out_<cfg>/
  --mode device  : cloud SdkCompiler build -> compile_out/artifact_<cfg>.json

<cfg> is the config-file basename, so concurrent configs never collide.
"""
import argparse
import json
import os
import subprocess
import time


def derive_params(cj):
    P = cj["P"]
    return {
        "P": P,
        "dim_p_pe": cj["dim"] // P,
        "pes_p_head": P // cj["n_heads"],
        "pes_p_kv_head": P // cj["n_kv_heads"],
        "head_dim_p_pe": cj["head_dim"] // P,
        "seq_len_p_pe": cj["seq_len"] // P,
        "ffn_dim_p_pe": cj["ffn_dim"] // P,
    }


def main():
    ap = argparse.ArgumentParser(description="Compile Prefill (WSE-3, SDK 2.10)")
    ap.add_argument("--mode", choices=["sim", "device"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cand-addr", type=int, default=0x8000,
                    help="Byte address for .cand_rope / .slot_code. Relinking at a\ndifferent address and diffing rope_body's bytes is the test for whether\nits branches are PC-relative or absolute.")
    args = ap.parse_args()

    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    with open(args.config, encoding="utf-8") as f:
        cj = json.load(f)
    d = derive_params(cj)
    P = d["P"]
    params = ",".join(f"{k}:{v}" for k, v in d.items())

    print("Start compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)

    if args.mode == "sim":
        out_dir = f"out_{cfg_name}"
        if args.cand_addr != 0x8000:
            out_dir += f"_cand{args.cand_addr:x}"
        subprocess.run(["rm", "-rf", out_dir], check=True)
        # Code-overlay pinning. Every global rope_body names must sit at the SAME
        # byte address on the compute PEs (prefill.csl) and the storage PE
        # (storage.csl), or the transplanted bytes read the wrong memory. The
        # code sections share 0x8000 deliberately: the two programs are linked
        # separately and never both contain .cand_rope and .slot_code, and
        # matching them protects rope_body's backward loop branch.
        # Free window on both PEs is 0x4c70..0xf680; 128 B spacing leaves room
        # for these to grow with dim_p_pe.
        sections = ",".join([
            ".rope_ptr_src:20480",      # 0x5000
            ".rope_freqs_cos:20608",    # 0x5080
            ".rope_freqs_sin:20736",    # 0x5100
            ".rope_cos_val:20864",      # 0x5180
            ".rope_sin_val:20992",      # 0x5200
            ".rope_dsd_1:21120",        # 0x5280
            ".rope_dsd_2:21248",        # 0x5300
            ".rope_tmp_1:21376",        # 0x5380
            ".rope_tmp_2:21504",        # 0x5400
            ".rope_tmp_3:21632",        # 0x5480
            ".rope_tmp_4:21760",        # 0x5500
            ".rope_tmp_1_dsd:21888",    # 0x5580
            ".rope_tmp_2_dsd:22016",    # 0x5600
            ".rope_tmp_3_dsd:22144",    # 0x5680
            ".rope_tmp_4_dsd:22272",    # 0x5700
            f".cand_rope:{args.cand_addr}",
            f".slot_code:{args.cand_addr}",
        ])
        cmd = [
            "cslc", "--arch=wse3", "./src/layout.csl",
            f"--fabric-dims={P + 8},{P + 2}", "--fabric-offsets=4,1",
            f"--params={params}", "-o", out_dir, "--memcpy", "--channels", "1",
            f"--link-section-start-address-bytes={sections}",
        ]
        subprocess.run(cmd, check=True)
    else:
        from cerebras.sdk.client import SdkCompiler
        os.makedirs("compile_out", exist_ok=True)
        options = (
            f"--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 "
            f"-o out --memcpy --channels=4 --params={params}"
        )
        with SdkCompiler(resource_cpu=48000, resource_mem=64 << 30, disable_version_check=True) as compiler:
            artifact_id = compiler.compile(
                app_path="src", csl_main="layout.csl", options=options, out_path="compile_out",
            )
        with open(f"compile_out/artifact_{cfg_name}.json", "w", encoding="utf-8") as f:
            json.dump({"artifact_id": artifact_id}, f)

    print("End compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)


if __name__ == "__main__":
    main()
