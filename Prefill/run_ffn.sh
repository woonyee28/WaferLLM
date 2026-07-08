set -e
# Usage: ./run_ffn.sh <ffn_config.json>   (FFN-only program at P=256, real WSE-3 via cloud compiler)
CONFIG=$1
if [ -z "$CONFIG" ]; then
    echo "usage: ./run_ffn.sh <ffn_config.json>"
    exit 1
fi

python compile.py --mode device --config "$CONFIG"
python launch_ffn.py --config "$CONFIG"
