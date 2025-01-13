NUM_SPLITS=$1
SPLIT=$2

# Validate inputs
if [[ -z "$NUM_SPLITS" || -z "$SPLIT" ]]; then
    echo "Usage: $0 <num_splits> <split>"
    exit 1
fi

# Run the Python script with the specified arguments
python inference.py \
    --dataset_path "/input/jieyuz2/weikaih/data/tmp_data/meta_sa/sa_000000" \
    --output_data_path "/input/jieyuz2/weikaih/data/ic_light_result" \
    --num_splits "$NUM_SPLITS" \
    --split "$SPLIT" \
    --index_json_path "/input/jieyuz2/weikaih/improve_segment/IC-Light/image_index_1_12.json"