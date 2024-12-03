#!/bin/bash

# establishing all of the variables
SCRIPT_DIR=$(pwd) # get current dir of scripts 
BASE_DIR=$(dirname "$SCRIPT_DIR") # get base dir
EVAL_SCRIPT="$BASE_DIR/quantization_calibration.py" # eval script path 

# quantization run 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore "$EVAL_SCRIPT" \
    --model_name_or_path mistralai/Ministral-8B-Instruct-2410 \
    --huggingface_dataset_path_or_name HuggingFaceH4/ultrachat_200k \
    --save_dir ./quantized_models/ \
    --max_sequence_length 2048 \
    --num_calibration_samples 512 \
    --seed 10 