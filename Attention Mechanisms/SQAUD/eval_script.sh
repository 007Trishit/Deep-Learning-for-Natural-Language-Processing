#!/bin/bash

# Configuration
OUTPUT_DIR="results/full_training"
mkdir -p "$OUTPUT_DIR/v1"
mkdir -p "$OUTPUT_DIR/v2"

# Array to store model paths for each architecture
declare -A model_paths=(
    ["roberta"]="full_models/roberta/1.1/roberta-base/checkpoint-*"
    ["roberta2"]="full_models/roberta/2.0/roberta-base/checkpoint-*"
    ["deberta"]="full_models/deberta-v3/1.1/deberta-v3-base/checkpoint-*"
    ["deberta2"]="full_models/deberta-v3/2.0/deberta-v3-base/checkpoint-*"
    ["bigbird"]="full_models/bigbird/1.1/bigbird-roberta-base/checkpoint-*"
    ["bigbird2"]="full_models/bigbird/2.0/bigbird-roberta-base/checkpoint-*"
    ["longformer"]="full_models/longformer/1.1/longformer-base-4096/checkpoint-*"
    ["longformer2"]="full_models/longformer/2.0/longformer-base-4096/checkpoint-*"
    ["roformer"]="full_models/roformer/1.1/roformer-base/checkpoint-*"
    ["roformer2"]="full_models/roformer/2.0/roformer-base/checkpoint-*"
    ["nystromformer"]="full_models/nystromformer/1.1/nystromformer-base/checkpoint-*"
    ["nystromformer2"]="full_models/nystromformer/2.0/nystromformer-base/checkpoint-*"
    ["yoso"]="full_models/yoso/1.1/yoso-base/checkpoint-*"
    ["yoso2"]="full_models/yoso/2.0/yoso-base/checkpoint-*"
    ["mega"]="full_models/mega/1.1/mega-base/checkpoint-*"
    ["mega2"]="full_models/mega/2.0/mega-base/checkpoint-*"
)

# Array to store child PIDs
declare -A pids

# Maximum number of parallel evaluations
MAX_PARALLEL=4
running_jobs=0

# Function to kill all child processes
cleanup() {
    echo "Interruption received. Terminating all processes..."
    for pid in "${!pids[@]}"; do
        echo "Killing process $pid (${pids[$pid]})"
        kill -TERM "$pid" 2>/dev/null
    done
    wait
    echo "All processes terminated."
    exit 1
}

# Set up trap for clean termination
trap cleanup SIGINT SIGTERM

# Setup Conda environment
eval "$(conda shell.bash hook)"
conda activate dl-nlp

# Function to run evaluation for a single model
run_evaluation() {
    local model_path=$1
    local squad_version=$2
    local model_name
    model_name="$(basename "$(dirname "$(dirname "$model_path")")")"
    local checkpoint
    checkpoint=$(basename "$model_path")
    local output_file="${OUTPUT_DIR}/${squad_version}/${model_name}_${checkpoint}_results.txt"
    local gpu_id=$3
    
    echo "Evaluating model: $model_path on SQuAD $squad_version (GPU $gpu_id)"
    
    # Set specific GPU for this evaluation
    CUDA_VISIBLE_DEVICES=$gpu_id python evaluate_squad.py \
        --model_path "$model_path" \
        --squad_version "$squad_version" \
        > "$output_file" 2>&1 &
    
    local pid=$!
    pids[$pid]="${model_name}_${squad_version}"
    ((running_jobs++))
    
    echo "Started evaluation process $pid for ${model_name} (SQuAD ${squad_version}) on GPU $gpu_id"
}

# Function to wait for available slot
wait_for_slot() {
    while [ $running_jobs -ge $MAX_PARALLEL ]; do
        for pid in "${!pids[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "Process $pid (${pids[$pid]}) completed"
                unset "pids[$pid]"
                ((running_jobs--))
            fi
        done < <(find "${OUTPUT_DIR}/${squad_version}" -type f -name "*_results.txt")
        sleep 5
    done
}

# Function to get next available GPU
get_next_gpu() {
    echo $((running_jobs % MAX_PARALLEL))
}

# Main evaluation loop
echo "Starting evaluations at $(date)"

# Evaluate all models
for model_key in "${!model_paths[@]}"; do
    # Get the latest checkpoint for each model using find
    latest_checkpoint=$(find "${model_paths[$model_key]%/*}" -maxdepth 1 -type d -name "checkpoint-*" -printf "%f\n" 2>/dev/null | sort -V | tail -n1)
    if [ -n "$latest_checkpoint" ]; then
        latest_checkpoint="${model_paths[$model_key]%/*}/$latest_checkpoint"
    fi
    
    if [ -z "$latest_checkpoint" ]; then
        echo "Warning: No checkpoint found for $model_key at ${model_paths[$model_key]}"
        continue
    fi
    
    # Determine SQuAD version
    if [[ $model_key == *"2"* ]]; then
        squad_version="v2"
    else
        squad_version="v1"
    fi
    
    wait_for_slot
    gpu_id=$(get_next_gpu)
    run_evaluation "$latest_checkpoint" "$squad_version" "$gpu_id"
done

# Wait for remaining evaluations to complete
echo "Waiting for remaining evaluations to finish..."
wait

# Generate summary report
echo "Generating evaluation summary..."
{
    echo "Evaluation Summary"
    echo "================="
    echo "Completed at: $(date)"
    echo ""
    echo "Results:"
    
    # Process results for each model
    for squad_version in "v1" "v2"; do
        echo ""
        echo "SQuAD $squad_version Results:"
        echo "------------------------"
        
        # Find all result files for this version using find
        while IFS= read -r result_file; do
            if [ -f "$result_file" ]; then
                model_name=$(basename "$result_file" | cut -d'_' -f1)
                
                # Extract metrics from result file
                exact_match=$(grep "Exact Match" "$result_file" | tail -n1 | awk '{print $NF}')
                f1_score=$(grep "F1 Score" "$result_file" | tail -n1 | awk '{print $NF}')
                
                echo "$model_name:"
                echo "  Exact Match: $exact_match"
                echo "  F1 Score: $f1_score"
            fi
        done
    done
} > "${OUTPUT_DIR}/evaluation_summary.txt"

echo "Evaluation summary saved to ${OUTPUT_DIR}/evaluation_summary.txt"

# Check for failed evaluations
failed_evals=0
for result_file in "$OUTPUT_DIR"/*/*.txt; do
    if grep -q "Error" "$result_file" || grep -q "Exception" "$result_file"; then
        echo "Error found in $result_file"
        ((failed_evals++))
    fi
done

echo "All evaluations completed at $(date)"
echo "Total failed evaluations: $failed_evals"
if [ $failed_evals -gt 0 ]; then
    echo "Check the logs in $OUTPUT_DIR for details on failed evaluations"
    exit 1
fi