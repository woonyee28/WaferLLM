set -e

# export SINGULARITYENV_SIMFABRIC_DEBUG=router
CONFIG=$1

if [ -z "$CONFIG" ]; then
    CONFIG="config.json"
fi

# if config.json exists
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
    echo "Use default test values."
    P=8
    GROUP_NUM=2
    BSZ=1
    DIM=64
    N_HEADS=1
    N_KV_HEADS=1
    HEAD_DIM=64
    MAX_SEQ_LEN=128
    PREFILL_LEN=64
    FFN_DIM=64
    FFN_ONLY=0
    ATTN_ONLY=0
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

echo "P: $P"
echo "BSZ: $BSZ"
echo "DIM: $DIM"
echo "N_HEADS: $N_HEADS"
echo "N_KV_HEADS: $N_KV_HEADS"
echo "HEAD_DIM: $HEAD_DIM"
echo "MAX_SEQ_LEN: $MAX_SEQ_LEN"
echo "PREFILL_LEN: $PREFILL_LEN"
echo "FFN_DIM: $FFN_DIM"

echo "GROUP_NUM: $GROUP_NUM"
echo "PE_NUM_PER_GROUP: $pe_num_p_group"
echo "ROOT_1ST_PHASE: $root_1st_phase"
echo "ROOT_2ND_PHASE: $root_2nd_phase"

# Validate: group_num must be a multiple of n_heads (for head-scoped reduce)
if [ $(( GROUP_NUM % N_HEADS )) -ne 0 ]; then
    echo "ERROR: group_num ($GROUP_NUM) must be a multiple of n_heads ($N_HEADS)"
    exit 1
fi

# Validate: group_num must be a multiple of n_kv_heads (for kv-head-scoped reduce)
if [ $(( GROUP_NUM % N_KV_HEADS )) -ne 0 ]; then
    echo "ERROR: group_num ($GROUP_NUM) must be a multiple of n_kv_heads ($N_KV_HEADS)"
    exit 1
fi

# Validate: prefill_len must be divisible by P
if [ $(( PREFILL_LEN % P )) -ne 0 ]; then
    echo "ERROR: prefill_len ($PREFILL_LEN) must be divisible by P ($P)"
    exit 1
fi

# Validate: max_seq_len must be divisible by P and >= prefill_len
if [ $(( MAX_SEQ_LEN % P )) -ne 0 ]; then
    echo "ERROR: max_seq_len ($MAX_SEQ_LEN) must be divisible by P ($P)"
    exit 1
fi

cslc --arch=wse3 ./src/layout.csl --fabric-dims="$FABRIC_W","$FABRIC_H" --fabric-offsets=4,1 \
    --params=P:"$P",bsz:"$BSZ",dim_p_pe:"$dim_p_pe",kv_dim_p_pe:"$kv_dim_p_pe",pes_p_head:"$pes_p_head",pes_p_kv_head:"$pes_p_kv_head",head_dim_p_pe:"$head_dim_p_pe",head_dim:"$HEAD_DIM",max_seq_len_p_pe:"$max_seq_len_p_pe",prefill_len_p_pe:"$prefill_len_p_pe",ffn_dim_p_pe:"$ffn_dim_p_pe",ffn_only:"$FFN_ONLY",attn_only:"$ATTN_ONLY",pe_num_p_group:"$pe_num_p_group",root_1st_phase:"$root_1st_phase",root_2nd_phase:"$root_2nd_phase" \
    -o out --memcpy --channels 1

cs_python launch_sim.py --config $CONFIG "${@:2}"

rm -rf simfab_traces
rm -rf wio_flows_tmpdir.*
rm wsjob-*.json
rm run_meta.json