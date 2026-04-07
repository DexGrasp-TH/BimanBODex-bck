#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/benchmark_synthesis_speed.sh --gpu GPU_ID --config <ROBOT>/<GRASP_TYPE>.yml [options]

Required:
  --gpu GPU_ID              CUDA device to use, e.g. 7
  --config CONFIG           Manipulation config, e.g. sim_shadow/tabletop_full.yml

Optional:
  --exp-name NAME           Base experiment name. Default: benchmark_synthesis
  --worlds "50 100 500"     Space-separated -w values to benchmark
  --start START             Override world.start
  --end END                 Override world.end
  --save-mode MODE          plan_batch_env.py save mode. Default: npy
  --save-data DATA          plan_batch_env.py save data. Default: all
  --output-dir DIR          Directory for logs and CSV. Default: outputs/speed_benchmark_results/<timestamp>
  -h, --help                Show this help

Example:
  ./scripts/benchmark_synthesis_speed.sh \
    --gpu 7 \
    --config sim_shadow/tabletop_full.yml \
    --exp-name speed_cmp \
    --start 0 \
    --end 500
EOF
}

GPU_ID=""
CONFIG=""
EXP_NAME="benchmark_synthesis"
WORLD_LIST="50 100 500"
START=""
END=""
SAVE_MODE="npy"
SAVE_DATA="all"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        --worlds)
            WORLD_LIST="$2"
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
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$GPU_ID" || -z "$CONFIG" ]]; then
    echo "--gpu and --config are required."
    usage
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="outputs/speed_benchmark_results/$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUTPUT_DIR"
RESULT_CSV="$OUTPUT_DIR/results.csv"

echo "config,parallel_world,exp_name,status,elapsed_seconds,log_file" > "$RESULT_CSV"

echo "Benchmark output: $OUTPUT_DIR"
echo "GPU: $GPU_ID"
echo "Config: $CONFIG"
echo "World sizes: $WORLD_LIST"
if [[ -n "$START" || -n "$END" ]]; then
    echo "World range override: start=${START:-<config>} end=${END:-<config>}"
fi
echo "================================"

for world in $WORLD_LIST; do
    run_exp_name="${EXP_NAME}_w${world}"
    log_file="$OUTPUT_DIR/${run_exp_name}.log"

    cmd=(
        python example_grasp/plan_batch_env.py
        -c "$CONFIG"
        -w "$world"
        -k
        --exp_name "$run_exp_name"
    )

    if [[ -n "$START" ]]; then
        cmd+=(--start "$START")
    fi

    if [[ -n "$END" ]]; then
        cmd+=(--end "$END")
    fi

    echo "Running benchmark for -w $world"
    echo "Command: CUDA_VISIBLE_DEVICES=$GPU_ID ${cmd[*]}"

    start_ns=$(date +%s%N)
    if CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" >"$log_file" 2>&1; then
        status="success"
    else
        status="failed"
    fi
    end_ns=$(date +%s%N)

    elapsed_seconds=$(awk "BEGIN {printf \"%.3f\", ($end_ns - $start_ns) / 1000000000}")
    echo "$CONFIG,$world,$run_exp_name,$status,$elapsed_seconds,$log_file" >> "$RESULT_CSV"

    echo "Status: $status"
    echo "Elapsed: ${elapsed_seconds}s"
    echo "Log: $log_file"
    echo "--------------------------------"
done

echo "Benchmark complete. Summary:"
cat "$RESULT_CSV"
