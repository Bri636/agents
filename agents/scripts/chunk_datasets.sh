#!/bin/bash

python -W ignore prepare_dataset.py \
    --dataset_name_or_path '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl' \
    --dataset_chunks_save_path '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/' \
    --num_chunks 4 \
    --max_size_per_chunk 256 \
    --seed 10 \
    --logging