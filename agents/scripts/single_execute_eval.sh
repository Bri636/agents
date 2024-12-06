#!/bin/bash
# DEVICE = 0
# establishing all of the variables
SCRIPT_DIR=$(pwd) # Get the current directory of scripts
BASE_DIR=$(dirname "$SCRIPT_DIR") # Get the base directory
# /lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/GSM_max_size-1024_num_chunks-4/
DATASET_DIR="$BASE_DIR/data/GSM_max_size-1024_num_chunks-4/"
DATASETS=("$DATASET_DIR"/*.jsonl) # array of dataset names 
EVAL_SCRIPT="$BASE_DIR/eval.py" # eval script path 
LOG_DIR="$BASE_DIR/log_files/"

CUDA_VISIBLE_DEVICES=0 python "$EVAL_SCRIPT" \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --master_config_path "$BASE_DIR/config_files/single_config.yaml" \
    --num_samples 64 \
    --batch_size 16 \
    --strategy mutate_mcts_world_model \
    --dtype bfloat16 \
    --logging_save_path ./single_gsm_outputs_device_${CUDA_VISIBLE_DEVICES}_mutate.log \
    --dataset_path /lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl
