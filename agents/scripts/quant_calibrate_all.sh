#!/bin/bash
# I Choose three models lmao
declare -a MODEL_NAMES=("google/gemma-2-2b-it" "mistralai/Ministral-8B-Instruct-2410")
# "meta-llama/Meta-Llama-3-8B-Instruct" 
# Iterate over each model
for model in "${MODEL_NAMES[@]}"; do
    echo "Starting quantization for model: $model"
    
    python quantization_calibration.py \
        --model_name_or_path "$model" \
        --huggingface_dataset_path_or_name "HuggingFaceH4/ultrachat_200k" \
        --save_dir "./quantized_models/" \
        --max_sequence_length 2048 \
        --num_calibration_samples 512 \
        --seed 10
    
    echo "Finished quantization for model: $model"
    echo "---------------------------------------------"
done
