#!/bin/bash

# Common Benchmark Parameters
BASE_RESULTS_FOLDER="benchmarking_results/classification/$(date +%Y%m%d_%H%M%S)" # Unique folder for this batch of runs
BASE_DATASET_FOLDER="data"

# Ensure Python's output is not buffered if run through a pipe or tee
export PYTHONUNBUFFERED=1

# Create the base results folder for this run
mkdir -p "$BASE_RESULTS_FOLDER"
echo "Results will be saved in $BASE_RESULTS_FOLDER"
echo "Running benchmarks with common settings:"
echo "  BASE_DATASET_FOLDER: $BASE_DATASET_FOLDER"
echo "-----------------------------------------------------"


# --- Define Model Configurations ---
# Format: "model_name_or_path;normalize_flag;prefix_function_name;trust_remote_code_flag"
# normalize_flag: "true" or "false" (will be converted to --normalize-embeddings if true)
# trust_remote_code_flag: "true" or "false" (will be converted to --trust-remote-code if true)

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
    "Alibaba-NLP/gte-multilingual-base;true;none"
    "mixedbread-ai/deepset-mxbai-embed-de-large-v1;false;none"
    "nomic-ai/nomic-embed-text-v2-moe;false;none"
    "Snowflake/snowflake-arctic-embed-l-v2.0;false;query"
    "shibing624/text2vec-base-multilingual;false;none"
)

# Loop through configurations and run benchmarks
for config in "${model_configs[@]}"; do
    IFS=';' read -r model_name normalize_str prefix_name <<< "$config"

    echo ""
    echo "--- Starting benchmark for: $model_name (Normalize: $normalize_str, Prefix: $prefix_name, Trust Remote Code: $trust_remote_code_str) ---"

    normalize_arg=""
    if [ "$normalize_str" == "true" ]; then
        normalize_arg="--normalize-embeddings"
    fi

    # Construct the command
    # Use python3 explicitly if that's your environment, or just python
    # Add tee to log stdout/stderr to a file per model and also print to console
    log_file_path="$BASE_RESULTS_FOLDER/${model_name//\//_}_${prefix_name}.log"

    python3 -m services.benchmark_classification.service \
        --model-name "$model_name" \
        $normalize_arg \
        --prefix-function-name "$prefix_name" \
        --base-results-folder "$BASE_RESULTS_FOLDER" \
        --base-dataset-folder "$BASE_DATASET_FOLDER" | tee "$log_file_path"

    echo "--- Finished benchmark for: $model_name ---"
    echo "Log saved to: $log_file_path"

    # Optional: Short delay between runs if needed, e.g. to let system cool down
    # echo "Pausing for 10 seconds..."
    # sleep 10
done

echo ""
echo "-----------------------------------------------------"
echo "All benchmarks completed. Results are in $BASE_RESULTS_FOLDER"