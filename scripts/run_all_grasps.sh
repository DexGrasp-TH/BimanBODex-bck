#!/bin/bash

# Run grasp planning for all robot/grasp_type combinations.
# Usage: ./scripts/run_all_grasps.sh [EXP_NAME] [GPU_ID] [NUM_PARALLEL_ENV] [START] [END]
# START and END optionally override world.start and world.end from each manipulation YAML.
# Examples:
#   ./scripts/run_all_grasps.sh minitest 0 10
#   ./scripts/run_all_grasps.sh minitest 0 10 0 10

EXP_NAME=${1:-"default"}
GPU_ID=${2:-7}
NUM_PARALLEL_ENV=${3:-10}
START=${4:-}
END=${5:-}

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
