#!/bin/bash

# establishing all of the variables
SCRIPT_DIR=$(pwd) # get current dir of scripts 
BASE_DIR=$(dirname "$SCRIPT_DIR") # get base dir
DATASET_DIR="$BASE_DIR/data/max_size-2000_num_chunks-4" # dataset dir for dataset chunks
DATASETS=("$DATASET_DIR"/*.jsonl) # array of dataset names 
EVAL_SCRIPT="$BASE_DIR/eval.py" # eval script path 

CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --master_config_path /lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/single_config.yaml \
    --num_samples 1000 \
    --batch_size 64 \
    --strategy mcts_world_model \
    --dtype bfloat16 \
    --logging_save_path ./single_gsm_outputs_device_${CUDA_VISIBLE_DEVICES}.log \
    --dataset_path /lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl 
