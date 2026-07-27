set -e

# Compile the Decode FFN stage (ffn_only=1, P=128) for $CONFIG, then validate resid_post against
# the transformer_lens oracle via launch_ffn_verify.py. Mirrors run_verify.sh's compile step.
# Extra args after the config are forwarded (e.g. --input csl_decode_zmid.npy --pos 64 --cmaddr ...).
CONFIG=$1

if [ -z "$CONFIG" ]; then
    CONFIG="model_config/llama8B_block0_ffn_p128.json"
fi

if [ -f $CONFIG ]; then
    echo "Use config values from $CONFIG."
    P=$(jq -r '.P' $CONFIG)
    GROUP_NUM=$(jq -r '.group_num' $CONFIG)
    BSZ=$(jq -r '.bsz' $CONFIG)
    DIM=$(jq -r '.dim' $CONFIG)
    N_HEADS=$(jq -r '.n_heads' $CONFIG)
    N_KV_HEADS=$(jq -r '.n_kv_heads' $CONFIG)
    HEAD_DIM=$(jq -r '.head_dim' $CONFIG)
    MAX_SEQ_LEN=$(jq -r '.max_seq_len' $CONFIG)
    PREFILL_LEN=$(jq -r '.prefill_len' $CONFIG)
    FFN_DIM=$(jq -r '.ffn_dim' $CONFIG)
    FFN_ONLY=$(jq -r '.ffn_only // 0' $CONFIG)
    ATTN_ONLY=$(jq -r '.attn_only // 0' $CONFIG)
else
    echo "ERROR: config $CONFIG not found"; exit 1
fi

FABRIC_W=$(($P + 7))
FABRIC_H=$(($P + 2))

dim_p_pe=$(($DIM / $P))
kv_dim_p_pe=$((($N_KV_HEADS * $HEAD_DIM) / $P))
pes_p_head=$(($P / $N_HEADS))
pes_p_kv_head=$(($P / $N_KV_HEADS))
head_dim_p_pe=$(($HEAD_DIM / $P))
max_seq_len_p_pe=$(($MAX_SEQ_LEN / $P))
prefill_len_p_pe=$(($PREFILL_LEN / $P))
ffn_dim_p_pe=$(($FFN_DIM / $P))
pe_num_p_group=$(($P / $GROUP_NUM))

root_1st_phase=$((pe_num_p_group / 2))
root_2nd_phase=$(((($GROUP_NUM / 2) * pe_num_p_group) + root_1st_phase))

echo "P=$P GROUP_NUM=$GROUP_NUM PE_NUM_PER_GROUP=$pe_num_p_group FFN_ONLY=$FFN_ONLY"
echo "DIM=$DIM N_HEADS=$N_HEADS N_KV_HEADS=$N_KV_HEADS HEAD_DIM=$HEAD_DIM FFN_DIM=$FFN_DIM"

if [ "$FFN_ONLY" -ne 1 ]; then
    echo "ERROR: run_ffn_verify.sh expects ffn_only=1 config (got $FFN_ONLY)"; exit 1
fi
if [ $(( GROUP_NUM % N_HEADS )) -ne 0 ]; then
    echo "ERROR: group_num ($GROUP_NUM) must be a multiple of n_heads ($N_HEADS)"; exit 1
fi
if [ $(( GROUP_NUM % N_KV_HEADS )) -ne 0 ]; then
    echo "ERROR: group_num ($GROUP_NUM) must be a multiple of n_kv_heads ($N_KV_HEADS)"; exit 1
fi
if [ $(( MAX_SEQ_LEN % P )) -ne 0 ]; then
    echo "ERROR: max_seq_len ($MAX_SEQ_LEN) must be divisible by P ($P)"; exit 1
fi

cslc --arch=wse3 ./src/layout.csl --fabric-dims="$FABRIC_W","$FABRIC_H" --fabric-offsets=4,1 \
    --params=P:"$P",bsz:"$BSZ",dim_p_pe:"$dim_p_pe",kv_dim_p_pe:"$kv_dim_p_pe",pes_p_head:"$pes_p_head",pes_p_kv_head:"$pes_p_kv_head",head_dim_p_pe:"$head_dim_p_pe",head_dim:"$HEAD_DIM",max_seq_len_p_pe:"$max_seq_len_p_pe",prefill_len_p_pe:"$prefill_len_p_pe",ffn_dim_p_pe:"$ffn_dim_p_pe",ffn_only:"$FFN_ONLY",attn_only:"$ATTN_ONLY",pe_num_p_group:"$pe_num_p_group",root_1st_phase:"$root_1st_phase",root_2nd_phase:"$root_2nd_phase" \
    -o out --memcpy --channels 1

cs_python launch_ffn_verify.py --config $CONFIG "${@:2}"

rm -rf simfab_traces
rm -rf wio_flows_tmpdir.*
rm -f wsjob-*.json
rm -f run_meta.json
