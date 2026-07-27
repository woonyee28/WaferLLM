set -e

# Disaggregated end-to-end Decode block-0 validation vs transformer_lens, appliance/cluster device
# flow (cloud compile; NO cmaddr). Two stages at different mesh sizes, chained through the DEVICE
# Z_mid (a true end-to-end layer, not two independently oracle-fed stages):
#
#   Stage 1  ATTENTION  P=64   (attn_only=1)   resid_pre -> Z_mid   (reports Z_pre, Z_mid)
#   Stage 2  FFN         P=128  (ffn_only=1)    Z_mid     -> Z_post  (reports Z_post)
#
# The two artifacts have DIFFERENT cfg names, so compile_out/ holds both. Each stage's cloud compile
# is CACHED (skipped if the artifact exists) — so re-runs launch instantly. Flags are forwarded to
# BOTH stages:  --simulator (appliance sim),  --force-compile (rebuild both artifacts).
#
# Usage:
#   bash run_layer_verify.sh                    # real WSE-3, reuse cached artifacts
#   bash run_layer_verify.sh --force-compile    # rebuild both, then run
#   bash run_layer_verify.sh --simulator        # appliance simulator
#
# Prereq: pytorch/oracle_decode/ generated (>= 65 tokens). From the repo root:
#   python pytorch/baseline_decode.py

ATTN_CONFIG=${ATTN_CONFIG:-model_config/llama8B_block0_p64_attn.json}
FFN_CONFIG=${FFN_CONFIG:-model_config/llama8B_block0_ffn_p128.json}

POS=$(jq -r '.prefill_len' $ATTN_CONFIG)
ZMID=csl_decode_zmid.npy

echo "======================================================================"
echo "  STAGE 1 / 2  —  ATTENTION  ($ATTN_CONFIG)  ->  Z_pre, Z_mid"
echo "======================================================================"
bash run_verify.sh "$ATTN_CONFIG" --save-zmid "$ZMID" "$@"

echo ""
echo "======================================================================"
echo "  STAGE 2 / 2  —  FFN  ($FFN_CONFIG)  fed DEVICE Z_mid  ->  Z_post"
echo "======================================================================"
bash run_ffn_verify.sh "$FFN_CONFIG" --input "$ZMID" --pos "$POS" "$@"

echo ""
echo "======================================================================"
echo "  DONE — Z_pre & Z_mid from stage 1, Z_post from stage 2 (above)."
echo "======================================================================"
