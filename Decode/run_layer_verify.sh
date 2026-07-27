set -e

# Disaggregated, end-to-end Decode block-0 validation vs transformer_lens, as TWO device launches
# at different mesh sizes (the full block does not fit per-PE at any single P):
#
#   Stage 1  ATTENTION  P=64   (ffn_only=0)   resid_pre --> Z_mid   (reports Z_pre, Z_mid)
#   Stage 2  FFN         P=128  (ffn_only=1)   Z_mid     --> Z_post  (reports Z_post)
#
# The device Z_mid from stage 1 (csl_decode_zmid.npy) is threaded into stage 2 (--input), so this
# is a true end-to-end layer, not two independently oracle-fed stages. Each cslc build overwrites
# `out`, which is safe because stage 1's device run finishes before stage 2 compiles.
#
# Usage:
#   bash run_layer_verify.sh                 # simulator
#   bash run_layer_verify.sh <CM_ADDRESS>    # WSE-3 hardware
#
# Prereq: pytorch/oracle_decode/ generated with a prompt of >= prefill_len+1 tokens (>=65 for P=64):
#   singularity exec --nv pytorch.sif python3 ../../pytorch/baseline_decode.py --prompt "..."

ATTN_CONFIG=${ATTN_CONFIG:-model_config/llama8B_block0_p64_attn.json}
FFN_CONFIG=${FFN_CONFIG:-model_config/llama8B_block0_ffn_p128.json}

CM=$1
if [ -n "$CM" ]; then CMARG="--cmaddr $CM"; else CMARG=""; fi

POS=$(jq -r '.prefill_len' $ATTN_CONFIG)
ZMID=csl_decode_zmid.npy

echo "======================================================================"
echo "  STAGE 1 / 2  —  ATTENTION  ($ATTN_CONFIG)  ->  Z_pre, Z_mid"
echo "======================================================================"
bash run_verify.sh $ATTN_CONFIG $CMARG --save-zmid $ZMID

echo ""
echo "======================================================================"
echo "  STAGE 2 / 2  —  FFN  ($FFN_CONFIG)  fed DEVICE Z_mid  ->  Z_post"
echo "======================================================================"
bash run_ffn_verify.sh $FFN_CONFIG --input $ZMID --pos $POS $CMARG

echo ""
echo "======================================================================"
echo "  DONE — Z_pre & Z_mid from stage 1, Z_post from stage 2 (above)."
echo "======================================================================"
