#!/bin/bash

# Common Benchmark Parameters
NUM_SENTENCES=15_000 # Adjusted for potentially faster runs during testing, increase for full benchmarks
# NUM_SENTENCES=10000 # Smaller for quick tests
BATCH_SIZE=64
TARGET_DEVICE="cpu" # or "cpu" or "mps"
BASE_RESULTS_FOLDER="benchmarking_results/performance/${TARGET_DEVICE}-m1-pro/$(date +%Y%m%d_%H%M%S)"
TEST_SENTENCE="This benchmark evaluates the inference speed and memory footprint of various sentence embedding models when processing a large volume of repeated textual data on different hardware configurations."
WARMUP_RUNS=256

# Ensure Python's output is not buffered if run through a pipe or tee
export PYTHONUNBUFFERED=1

# Create the base results folder for this run
mkdir -p "$BASE_RESULTS_FOLDER"
echo "Results will be saved in $BASE_RESULTS_FOLDER"
echo "Running benchmarks with common settings:"
echo "  NUM_SENTENCES: $NUM_SENTENCES"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  TARGET_DEVICE: $TARGET_DEVICE"
echo "  TEST_SENTENCE: \"$TEST_SENTENCE\""
echo "  WARMUP_RUNS: $WARMUP_RUNS"
echo "-----------------------------------------------------"


# --- Define Model Configurations ---
# Format: "model_name_or_path;normalize_flag;prefix_function_name"
# normalize_flag: "true" or "false" (will be converted to --normalize-embeddings if true)

declare -a model_configs=(
    "sentence-transformers/distiluse-base-multilingual-cased-v1;false;none"
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2;false;none"
    "sentence-transformers/distiluse-base-multilingual-cased-v2;false;none"
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2;false;none"
    "intfloat/multilingual-e5-large;true;query"
    "intfloat/multilingual-e5-large-instruct;true;e5_large_instruct"
    "intfloat/multilingual-e5-base;true;query"
    "intfloat/multilingual-e5-small;true;query"
    "ibm-granite/granite-embedding-278m-multilingual;false;none"
    "ibm-granite/granite-embedding-107m-multilingual;false;none"
    "mixedbread-ai/deepset-mxbai-embed-de-large-v1;false;none"
    "nomic-ai/nomic-embed-text-v2-moe;false;none"
    "shibing624/text2vec-base-multilingual;false;none"
)

# Loop through configurations and run benchmarks
for config in "${model_configs[@]}"; do
    IFS=';' read -r model_name normalize_str prefix_name <<< "$config"

    echo ""
    echo "--- Starting benchmark for: $model_name (Normalize: $normalize_str, Prefix: $prefix_name) ---"

    normalize_arg=""
    if [ "$normalize_str" == "true" ]; then
        normalize_arg="--normalize-embeddings"
    fi

    # Construct the command
    # Use python3 explicitly if that's your environment, or just python
    # Add tee to log stdout/stderr to a file per model and also print to console
    log_file_path="$BASE_RESULTS_FOLDER/${model_name//\//_}_${prefix_name}_${TARGET_DEVICE//:/}.log"

    python3 services/benchmark_performance/service.py \
        --model-name "$model_name" \
        $normalize_arg \
        --prefix-function-name "$prefix_name" \
        --num-sentences "$NUM_SENTENCES" \
        --batch-size "$BATCH_SIZE" \
        --device "$TARGET_DEVICE" \
        --base-results-folder "$BASE_RESULTS_FOLDER" \
        --test-sentence "$TEST_SENTENCE" \
        --warmup-runs "$WARMUP_RUNS" | tee "$log_file_path"
    
    echo "--- Finished benchmark for: $model_name ---"
    echo "Log saved to: $log_file_path"
    
    # Optional: Short delay between runs if needed, e.g. to let system cool down
    # echo "Pausing for 10 seconds..."
    # sleep 10
done

echo ""
echo "-----------------------------------------------------"
echo "All benchmarks completed. Results are in $BASE_RESULTS_FOLDER"