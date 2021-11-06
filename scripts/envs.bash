set -e
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
output=$(python3 $SCRIPT_DIR/envs.py $*)
eval $output
