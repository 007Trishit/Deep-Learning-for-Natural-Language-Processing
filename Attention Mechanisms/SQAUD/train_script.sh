#!/bin/bash

# Configuration
MODELS=("roberta" "deberta" "bigbird" "longformer" "roformer" "nystromformer" "yoso" "mega")
SQUAD_VERSIONS=("1.1" "2.0")
MAX_PARALLEL_JOBS=8  # Adjust based on available GPUs
LOG_DIR="logs/full_training"
OUTPUT_DIR="outputs/full_training"
CUDA_DEVICES=(0 1 2 3 4 5 6 7)  # Adjust based on available GPUs

# Array to store child PIDs
declare -A pids
running_jobs=0

# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

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

# Function to get next available GPU
get_next_gpu() {
    echo "${CUDA_DEVICES[$((running_jobs % ${#CUDA_DEVICES[@]}))]}"
}

# Function to run a single training job
run_training() {
    local model=$1
    local squad_version=$2
    local gpu
    gpu=$(get_next_gpu)
    
    # Create descriptive names for logging
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="${LOG_DIR}/${model}_squad${squad_version}_${timestamp}.log"
    
    echo "Starting training for $model on SQuAD $squad_version (GPU $gpu)"
    
    # Run the training script with specific GPU
    CUDA_VISIBLE_DEVICES=$gpu python trainer.py \
        --model_type "$model" \
        --squad_version "$squad_version" \
        --output_dir "${OUTPUT_DIR}/${model}" \
        > "$log_file" 2>&1 &
    
    local pid=$!
    pids[$pid]="${model}_squad${squad_version}"
    echo "Started process $pid for ${model}_squad${squad_version} on GPU $gpu"
    ((running_jobs++))
}

# Function to wait for available slot
wait_for_slot() {
    while [ $running_jobs -ge $MAX_PARALLEL_JOBS ]; do
        for pid in "${!pids[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "Process $pid (${pids[$pid]}) completed"
                unset "pids[$pid]"
                ((running_jobs--))
            fi
        done
        sleep 5
    done
}

# Main training loop
echo "Starting training jobs at $(date)"
echo "Logging to $LOG_DIR"

for model in "${MODELS[@]}"; do
    for version in "${SQUAD_VERSIONS[@]}"; do
        wait_for_slot
        run_training "$model" "$version"
    done
done

# Wait for remaining jobs to complete
echo "Waiting for remaining jobs to finish..."
wait

# Check for failed jobs
failed_jobs=0
for log_file in "$LOG_DIR"/*.log; do
    if grep -q "Error" "$log_file" || grep -q "Exception" "$log_file"; then
        echo "Error found in $log_file"
        ((failed_jobs++))
    fi
done

echo "All training jobs completed at $(date)"
echo "Total failed jobs: $failed_jobs"
if [ $failed_jobs -gt 0 ]; then
    echo "Check the logs in $LOG_DIR for details on failed jobs"
    exit 1
fi

# Generate summary report
echo "Generating training summary..."
{
    echo "Training Summary"
    echo "================"
    echo "Completed at: $(date)"
    echo "Models trained: ${MODELS[*]}"
    echo "SQuAD versions: ${SQUAD_VERSIONS[*]}"
    echo ""
    echo "Results:"
    for model in "${MODELS[@]}"; do
        for version in "${SQUAD_VERSIONS[@]}"; do
            echo "- $model (SQuAD $version):"
            log_file=$(find "${LOG_DIR}" -type f -name "${model}_squad${version}_*.log" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)
            if [ -f "$log_file" ]; then
                eval_results=$(grep "Final evaluation results" "$log_file" | tail -n1)
                echo "  $eval_results"
            else
                echo "  No log file found"
            fi
        done
    done
} > "${OUTPUT_DIR}/training_summary.txt"

echo "Training summary saved to ${OUTPUT_DIR}/training_summary.txt"