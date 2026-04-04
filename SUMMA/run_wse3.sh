set -e

if command -v git >/dev/null 2>&1; then
    repo_root=$(git rev-parse --show-toplevel 2>/dev/null || true)
    if [ -n "$repo_root" ]; then
        git_commit=$(git -C "$repo_root" rev-parse --short=12 HEAD 2>/dev/null || echo "unknown")
        git_branch=$(git -C "$repo_root" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
        if git -C "$repo_root" diff --quiet --ignore-submodules HEAD -- 2>/dev/null; then
            git_dirty="clean"
        else
            git_dirty="dirty"
        fi
        echo "Git commit: $git_commit"
        echo "Git branch: $git_branch"
        echo "Git status: $git_dirty"
    fi
fi

P=$1
Mt=$(($2 / $1))
Kt=$(($3 / $1))
Nt=$(($4 / $1))

simulator=false

if [ -n "$5" ]; then
    simulator=$5
fi

echo "P=$P, M=$2, K=$3, N=$4, Mt=$Mt, Kt=$Kt, Nt=$Nt, simulator=$simulator"

python compile.py "$P" "$Mt" "$Kt" "$Nt" "$simulator"

if [ "$simulator" == "true" ]; then
    python launch_wse3.py --P "$1" --M "$2" --K "$3" --N "$4" --simulator
else
    python launch_wse3.py --P "$1" --M "$2" --K "$3" --N "$4"
fi
