#!/bin/bash

# Function to handle cleanup on error or termination
cleanup() {
    echo "Error occurred or script terminated. Cleaning up..."
    # Kill all background jobs
    kill 0
    exit 1
}

# Trap signals and errors to execute the cleanup function
trap cleanup SIGINT SIGTERM ERR

# Establishing all of the variables
SCRIPT_DIR=$(pwd) # Get the current directory of scripts
BASE_DIR=$(dirname "$SCRIPT_DIR") # Get the base directory
CUDA_DEVICES=(0 1 2 3) # Define the list of CUDA devices
DATASET_DIR="$BASE_DIR/data/max_size-2000_num_chunks-4" # Dataset directory for dataset chunks
DATASETS=("$DATASET_DIR"/*.jsonl) # Array of dataset paths
EVAL_SCRIPT="$BASE_DIR/eval.py" # Eval script path
LOG_DIR="$BASE_DIR/log_files/"

# Ensure the number of devices matches the number of datasets
if [[ ${#DATASETS[@]} -ne ${#CUDA_DEVICES[@]} ]]; then
    echo "Error: The number of dataset chunks (${#DATASETS[@]}) does not match the number of CUDA devices (${#CUDA_DEVICES[@]})."
    exit 1
fi

# Loop over CUDA devices and datasets in parallel
for i in "${!CUDA_DEVICES[@]}"; do
    DEVICE=${CUDA_DEVICES[$i]}
    DATASET=${DATASETS[$i]}
    echo "Launching eval.py on CUDA device $DEVICE with dataset chunk $DATASET"
    CUDA_VISIBLE_DEVICES=$DEVICE python -W ignore "$EVAL_SCRIPT" \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --master_config_path "$BASE_DIR/config_files/single_config.yaml" \
        --num_samples 500 \
        --batch_size 64 \
        --strategy mcts_world_model \
        --dtype bfloat16 \
        --logging_save_path "$LOG_DIR/single_gsm_outputs_device_${DEVICE}.log" \
        --dataset_path "$DATASET" &
    # Capture the PID of the process
    PIDS+=($!)
done

# Wait for all background processes to finish
wait
echo "All tasks completed or failed."
