set -e

# Attention stage (attn_only=1, P=64): cloud-compile + validate Z_pre/Z_mid vs the transformer_lens
# oracle (pytorch/oracle_decode/). Appliance/cluster device flow (no cmaddr) — mirrors Prefill's
# run_device.sh. Extra args after the config are forwarded to launch_verify.py
# (e.g. --save-zmid <path>, --simulator).
CONFIG=$1
if [ -z "$CONFIG" ]; then
    CONFIG="model_config/llama8B_block0_p64_attn.json"
fi

python compile.py --mode device --config "$CONFIG"
python launch_verify.py --config "$CONFIG" "${@:2}"
