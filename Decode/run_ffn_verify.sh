set -e

# FFN stage (ffn_only=1, P=128): cloud-compile + validate Z_post vs the transformer_lens oracle.
# Appliance/cluster device flow (no cmaddr) — mirrors Prefill's run_ffn.sh. Extra args after the
# config are forwarded to launch_ffn_verify.py (e.g. --input csl_decode_zmid.npy --pos 64 --simulator).
CONFIG=$1
if [ -z "$CONFIG" ]; then
    CONFIG="model_config/llama8B_block0_ffn_p128.json"
fi

python compile.py --mode device --config "$CONFIG"
python launch_ffn_verify.py --config "$CONFIG" "${@:2}"
