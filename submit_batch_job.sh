#!/bin/bash

# Total number of jobs to run
TOTAL_JOBS=128

DATASET_PATH="/input/jieyuz2/weikaih/data/mask2former_dataset/datasets/synthetic_data_v3_relight/train"
OUTPUT_DATA_PATH="/input/jieyuz2/weikaih/data/mask2former_dataset/datasets/synthetic_data_v3_relight/train" 
RECORD_PATH="/input/jieyuz2/weikaih/data/ic_light_logs/mask2former_relight_v2_2_7.json" 

# Iterate through each job index
for ((i=0; i<$TOTAL_JOBS; i++)); do
    echo "Submitting job $i/$((TOTAL_JOBS - 1))"
    
    gantry run --allow-dirty \
        --name "relightening_${i}_$((TOTAL_JOBS - 1))_2_7_v2" \
        --task-name "segment-data-relightening" \
        --gpus 1 \
        --budget ai2/prior \
        --workspace ai2/improve_segments \
        --cluster ai2/neptune-cirrascale \
        --cluster ai2/jupiter-cirrascale-2 \
        --cluster ai2/saturn-cirrascale \
        --priority low \
        --preemptible \
        --beaker-image 'jieyuz2/semantic-sam-training-v1.11' \
        --venv 'base' \
        --pip requirements.txt \
        --weka 'prior-default:/input' \
        --shared-memory='100GiB' \
        -- /bin/bash inference_1_13.sh $TOTAL_JOBS $i "$DATASET_PATH" "$OUTPUT_DATA_PATH" "$RECORD_PATH"
done

echo "All $TOTAL_JOBS jobs have been submitted."