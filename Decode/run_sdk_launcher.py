#!/usr/bin/env python3
"""
SdkLauncher script for Decode WSE-3-GQA module.

Dispatches a cslc-compiled artifact to the appliance via SdkLauncher
using a staging directory containing the compiled output, host script,
and config file.

Prerequisites:
    python compile.py <config.json> [simulator]   # produces out/ directory

Usage:
    python run_sdk_launcher.py --config model_config/gqa_test.json
    python run_sdk_launcher.py --config model_config/gqa_test.json --simulator
"""
import argparse
import json
import os
import shutil
import sys

from cerebras.sdk.client import SdkLauncher


def main():
    parser = argparse.ArgumentParser(
        description="SdkLauncher dispatch for Decode WSE-3-GQA"
    )
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to JSON config file"
    )
    parser.add_argument(
        "--simulator", action="store_true",
        help="Run in appliance simulator mode"
    )
    parser.add_argument(
        "--steps", type=int, default=1,
        help="Total decode steps to run (default 1)"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir("out"):
        print(
            "Error: compiled output directory 'out/' not found.\n"
            "Run compile.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    config_basename = os.path.basename(args.config)

    # Build staging directory with compiled output + host scripts
    staging_dir = "launcher_staging"
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)

    shutil.copytree("out", os.path.join(staging_dir, "out"))
    shutil.copy2("launch_sim.py", staging_dir)
    shutil.copy2(args.config, staging_dir)

    run_cmd = (
        f"cs_python launch_sim.py --config {config_basename} "
        f"--steps {args.steps} --cmaddr %CMADDR%"
    )

    print(f"=== Decode WSE-3-GQA: SdkLauncher Dispatch ===")
    print(f"Config       : {args.config}")
    print(f"Staging dir  : {staging_dir}")
    print(f"Simulator    : {args.simulator}")
    print(f"Run command  : {run_cmd}")
    print()

    with SdkLauncher(staging_dir, simulator=args.simulator,
                     disable_version_check=True) as launcher:
        print("Executing on appliance...")
        response = launcher.run(run_cmd)

    print("Appliance response:")
    print(response)

    # Clean up staging directory
    shutil.rmtree(staging_dir, ignore_errors=True)

    return response


if __name__ == "__main__":
    main()
