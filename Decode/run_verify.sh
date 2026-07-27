set -e

# Attention stage (attn_only=1, P=64): cloud-compile + validate Z_pre/Z_mid vs the transformer_lens
# oracle (pytorch/oracle_decode/). Appliance/cluster device flow (no cmaddr) — mirrors Prefill.
#
# CACHING: the (slow) cloud compile is SKIPPED when compile_out/artifact_<cfg>.json already exists.
# Pass --force-compile to rebuild. All other extra args are forwarded to launch_verify.py
# (e.g. --save-zmid <path>, --simulator).
#
# Usage: bash run_verify.sh [config.json] [--force-compile] [--simulator] [--save-zmid <path>]

# Strip --force-compile out of the args (the launcher's argparse would reject it); keep the rest.
FORCE=0
ARGS=()
for a in "$@"; do
    if [ "$a" = "--force-compile" ]; then FORCE=1; else ARGS+=("$a"); fi
done
set -- "${ARGS[@]}"

CONFIG=$1
if [ -z "$CONFIG" ]; then
    CONFIG="model_config/llama8B_block0_p64_attn.json"
fi

ARTIFACT="compile_out/artifact_$(basename "${CONFIG%.json}").json"
if [ "$FORCE" -eq 1 ] || [ ! -f "$ARTIFACT" ]; then
    python compile.py --mode device --config "$CONFIG"
else
    echo "[cache] reusing $ARTIFACT  (pass --force-compile to rebuild)"
fi

python launch_verify.py --config "$CONFIG" "${@:2}"
