#!/bin/bash

set -u

# Run grasp planning for all robot/grasp_type combinations.
# Usage:
#   ./scripts/run_all_grasps.sh --exp-name NAME --gpu GPU_ID --parallel-env N [--start START] [--end END]
# Examples:
#   ./scripts/run_all_grasps.sh --exp-name minitest --gpu 0 --parallel-env 10
#   ./scripts/run_all_grasps.sh --exp-name minitest --gpu 0 --parallel-env 10 --start 0 --end 10

EXP_NAME="default"
GPU_ID=7
NUM_PARALLEL_ENV=10
START=""
END=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --parallel-env)
            NUM_PARALLEL_ENV="$2"
            shift 2
            ;;
        --start)
            START="$2"
            shift 2
            ;;
        --end)
            END="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./scripts/run_all_grasps.sh --exp-name NAME --gpu GPU_ID --parallel-env N [--start START] [--end END]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage."
            exit 1
            ;;
    esac
done

# Define robot and grasp type combinations
declare -a CONFIGS=(
    "sim_shadow/tabletop_two"
    "sim_shadow/tabletop_three"
    "sim_shadow/tabletop_full"
    "sim_dual_dummy_arm_shadow/tabletop_three"
    "sim_dual_dummy_arm_shadow/tabletop_full"
)

echo "Starting grasp planning experiments..."
echo "Experiment Name: $EXP_NAME"
echo "GPU: $GPU_ID, Parallel Envs: $NUM_PARALLEL_ENV"
if [ -n "$START" ] || [ -n "$END" ]; then
    echo "World Range Override: start=${START:-<config>}, end=${END:-<config>}"
fi
echo "================================"

for config in "${CONFIGS[@]}"; do
    config_name=$(echo "$config" | tr '/' '_')

    echo ""
    echo "Running: $EXP_NAME"
    echo "Config: $config"
    echo "--------------------------------"

    cmd=(
        python example_grasp/plan_batch_env.py
        -c "$config.yml" \
        -w "$NUM_PARALLEL_ENV" \
        -k \
        --exp_name "$EXP_NAME"
    )

    if [ -n "$START" ]; then
        cmd+=(--start "$START")
    fi

    if [ -n "$END" ]; then
        cmd+=(--end "$END")
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID "${cmd[@]}"

    if [ $? -eq 0 ]; then
        echo "✓ Completed: ${config_name}/${EXP_NAME}"
    else
        echo "✗ Failed: ${config_name}/${EXP_NAME}"
    fi
done

echo ""
echo "================================"
echo "All experiments completed!"
