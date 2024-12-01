#!/bin/bash

# Set up logging
LOG_DIR="training_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/training_${TIMESTAMP}.log"

# Function to log messages
log_message() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" | tee -a "$MAIN_LOG"
}

# Function to check CUDA
check_cuda() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_message "ERROR: NVIDIA GPU driver not found"
        exit 1
    fi
    
    log_message "CUDA Status:"
    nvidia-smi | tee -a "$MAIN_LOG"
    
    local available_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    if [ "$available_gpus" -ne 8 ]; then
        log_message "WARNING: Expected 8 GPUs, found $available_gpus"
    fi
}

# Function to train a model version
train_model_version() {
    local script=$1
    local model_name=$2
    local version=$3
    local gpu=$4
    local log_file="$LOG_DIR/${model_name}_squad${version}_${TIMESTAMP}.log"
    
    log_message "Starting ${model_name} training for SQuAD ${version} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES=$gpu python3 $script --squad_version $version &> "$log_file" &
    
    # Return the PID
    echo $!
}

# Main execution
main() {
    log_message "Starting parallel training pipeline on all GPUs"
    check_cuda
    
    # Create directories
    mkdir -p saved_models runs
    
    # Define model configurations - each version gets 1 GPU
    # Format: "script:model_name:gpu_v1:gpu_v2"
    declare -A models=(
        ["roberta"]="roberta_trainer.py:RoBERTa:0:1"
        ["deberta"]="deberta_trainer.py:DeBERTa:2:3"
        ["bigbird"]="bb_trainer.py:BigBird:4:5"
        ["longformer"]="lf_trainer.py:Longformer:6:7"
    )
    
    # Store PIDs and model names
    declare -A pids
    declare -A model_versions
    
    # Launch all training processes
    for model_key in "${!models[@]}"; do
        IFS=":" read -r script model_name gpu_v1 gpu_v2 <<< "${models[$model_key]}"
        
        if [ -f "$script" ]; then
            # Launch SQuAD v1.1
            pid_v1=$(train_model_version "$script" "$model_name" "1.1" "$gpu_v1")
            pids[$pid_v1]=$pid_v1
            model_versions[$pid_v1]="${model_name} SQuAD v1.1"
            
            # Launch SQuAD v2.0
            pid_v2=$(train_model_version "$script" "$model_name" "2.0" "$gpu_v2")
            pids[$pid_v2]=$pid_v2
            model_versions[$pid_v2]="${model_name} SQuAD v2.0"
        else
            log_message "ERROR: Script $script not found"
        fi
    done
    
    # Monitor all processes
    failed_models=()
    for pid in "${!pids[@]}"; do
        if wait "$pid"; then
            log_message "✓ ${model_versions[$pid]} completed successfully"
        else
            log_message "✗ ${model_versions[$pid]} failed"
            failed_models+=("${model_versions[$pid]}")
        fi
    done
    
    # Print final summary
    log_message "\nTraining Pipeline Summary"
    log_message "========================"
    if [ ${#failed_models[@]} -eq 0 ]; then
        log_message "✓ All models trained successfully!"
    else
        log_message "✗ The following models failed:"
        for model in "${failed_models[@]}"; do
            log_message "  - $model"
        done
    fi
    
    log_message "\nFinal GPU Status:"
    nvidia-smi | tee -a "$MAIN_LOG"
    
    log_message "\nTotal training time: $SECONDS seconds"
}

# Execute main function
main