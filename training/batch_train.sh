#!/bin/bash
# Usage: ./batch_train.sh <config1> <run_name1> [<config2> <run_name2> ...]
# Example: ./batch_train.sh configs/per_annotator_config_small_user_history.json run_small \
#                           configs/per_annotator_config_large_user_history.json run_large

set -e

if [ $(( $# % 2 )) -ne 0 ] || [ $# -eq 0 ]; then
    echo "Usage: $0 <config1> <run_name1> [<config2> <run_name2> ...]"
    exit 1
fi

total=$(( $# / 2 ))
job=0

while [ $# -gt 0 ]; do
    config=$1
    run_name=$2
    shift 2
    job=$(( job + 1 ))

    echo "[$job/$total] Starting: $run_name (config: $config)"
    python train_per_annotator.py "$config" "$run_name"
    echo "[$job/$total] Done: $run_name"
done

echo "All $total jobs completed."
