import os
import time
import argparse
import json

from cerebras.sdk.client import SdkLauncher

import logging
from cerebras.appliance import logger

logging.basicConfig(level=logging.INFO)

out_path = "compile_out"

def parse_args():
    parser = argparse.ArgumentParser(description="SUMMA GEMM on WSE-3 via SdkLauncher")

    parser.add_argument("--P", required=True, type=int, help="PEs rectangle size: P x P")
    parser.add_argument("--M", required=True, type=int, help="Input context length")
    parser.add_argument("--K", required=True, type=int, help="Word vector dimension")
    parser.add_argument("--N", required=True, type=int, help="Output dimension")
    parser.add_argument("--simulator", action="store_true", help="Run in simulator mode")

    return parser.parse_args()

def main():
    args = parse_args()

    P = args.P
    M = args.M
    K = args.K
    N = args.N
    Mt = M // P
    Kt = K // P
    Nt = N // P

    with open(f"{out_path}/artifact_{P}_{Mt}_{Kt}_{Nt}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        artifact_id = data["artifact_id"]

    print("Start running: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)
    print(f"Running on simulator: {args.simulator}")

    if args.simulator:
        warmup, repeat = 2, 10
    else:
        warmup, repeat = 10, 100

    with SdkLauncher(artifact_id, simulator=args.simulator, disable_version_check=True) as launcher:
        # Stage launch_sim.py to the appliance
        script_dir = os.path.dirname(os.path.abspath(__file__))
        launcher.stage(os.path.join(script_dir, "launch_sim.py"))

        # Build cs_python command; %CMADDR% is auto-replaced by the appliance
        if args.simulator:
            cmd = (
                f"cs_python launch_sim.py"
                f" --name out"
                f" --P {P} --M {M} --K {K} --N {N}"
                f" --warmup {warmup} --repeat {repeat}"
                f" --perf_only"
            )
        else:
            cmd = (
                f"cs_python launch_sim.py"
                f" --name out --cmaddr %CMADDR%"
                f" --P {P} --M {M} --K {K} --N {N}"
                f" --warmup {warmup} --repeat {repeat}"
                f" --perf_only"
            )

        response = launcher.run(cmd)
        print(response)

    print("End running: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)

if __name__ == "__main__":
    main()
