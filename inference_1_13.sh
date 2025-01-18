#!/bin/bash

NUM_SPLITS=$1
SPLIT=$2
DATASET_PATH=$3
OUTPUT_DATA_PATH=$4
RECORD_PATH=$5

# Validate inputs
if [[ -z "$NUM_SPLITS" || -z "$SPLIT" || -z "$DATASET_PATH" || -z "$OUTPUT_DATA_PATH" ]]; then
    echo "Usage: $0 <num_splits> <split> <dataset_path> <output_data_path>"
    exit 1
fi

# Run the Python script with the specified arguments
python inference.py \
    --dataset_path "$DATASET_PATH" \
    --output_data_path "$OUTPUT_DATA_PATH" \
    --num_splits "$NUM_SPLITS" \
    --split "$SPLIT" \
    --index_json_path "/input/jieyuz2/weikaih/improve_segment/IC-Light/image_index_1_12.json" \
    --illuminate_prompts_path "/input/jieyuz2/weikaih/improve_segment/IC-Light/illumination_prompt.json" \
    --record_path "$RECORD_PATH"