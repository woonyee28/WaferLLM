set -e
# Usage: ./run_sim.sh <config.json> [launcher args, e.g. --replicate-bsz1]   (small configs only — sim cost scales with P*P)
CONFIG=$1
if [ -z "$CONFIG" ]; then
    echo "usage: ./run_sim.sh <config.json>"
    exit 1
fi

python compile.py --mode sim --config "$CONFIG"
shift
cs_python launch_sim.py --config "$CONFIG" "$@"
