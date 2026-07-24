set -e

CONFIG=$1

if [ -z "$CONFIG" ]; then
    CONFIG="config.json"
fi

simulator=false

if [ -n "$2" ]; then
    simulator=$2
fi

STEPS=1

if [ -n "$3" ]; then
    STEPS=$3
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
else
    echo "Error: config file not found: $CONFIG"
    exit 1
fi

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
echo "Simulator: $simulator"
echo "Steps: $STEPS"

# Validate: group_num must be a multiple of n_heads
if [ $(( GROUP_NUM % N_HEADS )) -ne 0 ]; then
    echo "ERROR: group_num ($GROUP_NUM) must be a multiple of n_heads ($N_HEADS)"
    exit 1
fi

# Validate: group_num must be a multiple of n_kv_heads
if [ $(( GROUP_NUM % N_KV_HEADS )) -ne 0 ]; then
    echo "ERROR: group_num ($GROUP_NUM) must be a multiple of n_kv_heads ($N_KV_HEADS)"
    exit 1
fi

# Step 1: Compile using cslc
python compile.py $CONFIG $simulator

# Step 2: Dispatch to appliance via SdkLauncher
if [ "$simulator" == "true" ]; then
    python run_sdk_launcher.py --config $CONFIG --simulator --steps $STEPS
else
    python run_sdk_launcher.py --config $CONFIG --steps $STEPS
fi
