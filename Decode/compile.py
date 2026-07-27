import os
import sys
import json
import time
import subprocess

def main():
    # Two interfaces:
    #   legacy : python compile.py <config.json> [simulator]      -> local cslc into out/
    #   device : python compile.py --config <cfg> --mode device   -> cloud SdkCompiler ->
    #            compile_out/artifact_<cfg>.json (Prefill-style). Appliance/cluster hosts
    #            (e.g. EIDF eidf002-cs3) have NO cmaddr, so this is the only device path.
    import argparse
    if any(a.startswith("--") for a in sys.argv[1:]):
        ap = argparse.ArgumentParser(description="Compile Decode (WSE-3, SDK 2.10)")
        ap.add_argument("--config", required=True)
        ap.add_argument("--mode", choices=["sim", "device"], default="sim")
        a = ap.parse_args()
        config_path, mode = a.config, a.mode
        simulator = (mode == "sim")
    else:
        if len(sys.argv) < 2:
            print("Usage: python compile.py <config.json> [simulator] | "
                  "--config <cfg> --mode {sim,device}", file=sys.stderr)
            sys.exit(1)
        config_path = sys.argv[1]
        simulator = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False
        mode = "sim"

    with open(config_path, "r", encoding="utf8") as f:
        config = json.load(f)

    P = config["P"]
    bsz = config["bsz"]
    group_num = config["group_num"]
    dim = config["dim"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    head_dim = config["head_dim"]
    max_seq_len = config["max_seq_len"]
    prefill_len = config["prefill_len"]
    ffn_dim = config["ffn_dim"]

    dim_p_pe = dim // P
    kv_dim_p_pe = (n_kv_heads * head_dim) // P
    pes_p_head = P // n_heads
    pes_p_kv_head = P // n_kv_heads
    head_dim_p_pe = head_dim // P
    max_seq_len_p_pe = max_seq_len // P
    prefill_len_p_pe = prefill_len // P
    ffn_dim_p_pe = ffn_dim // P
    pe_num_p_group = P // group_num
    root_1st_phase = pe_num_p_group // 2
    root_2nd_phase = (group_num // 2) * pe_num_p_group + root_1st_phase
    # Disaggregated decode: ffn_only=1 -> FFN stage; attn_only=1 -> attention stage stops at
    # resid_mid; both 0 (default) -> monolithic full block (backward compatible).
    ffn_only = config.get("ffn_only", 0)
    attn_only = config.get("attn_only", 0)

    params = (
        f"P:{P},bsz:{bsz},"
        f"dim_p_pe:{dim_p_pe},kv_dim_p_pe:{kv_dim_p_pe},"
        f"pes_p_head:{pes_p_head},pes_p_kv_head:{pes_p_kv_head},"
        f"head_dim_p_pe:{head_dim_p_pe},head_dim:{head_dim},"
        f"max_seq_len_p_pe:{max_seq_len_p_pe},prefill_len_p_pe:{prefill_len_p_pe},"
        f"ffn_dim_p_pe:{ffn_dim_p_pe},ffn_only:{ffn_only},attn_only:{attn_only},"
        f"pe_num_p_group:{pe_num_p_group},"
        f"root_1st_phase:{root_1st_phase},root_2nd_phase:{root_2nd_phase}"
    )

    cfg_name = os.path.splitext(os.path.basename(config_path))[0]
    print(f"Start compiling ({mode}): {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    if mode == "device":
        # Cloud compiler -> artifact_id; run later via cerebras.sdk.client.SdkRuntime(simulator=False).
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
        print(f"artifact -> compile_out/artifact_{cfg_name}.json", flush=True)
    else:
        if simulator:
            fabric_w, fabric_h, channels = P + 7, P + 2, 1
        else:
            fabric_w, fabric_h, channels = 762, 1172, 4
        cmd = (
            f"cslc --arch=wse3 ./src/layout.csl "
            f"--fabric-dims={fabric_w},{fabric_h} --fabric-offsets=4,1 "
            f"--params={params} -o out --memcpy --channels {channels}"
        )
        print(f"Command: {cmd}", flush=True)
        subprocess.run(cmd, shell=True, check=True)

    print(f"End compiling: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
