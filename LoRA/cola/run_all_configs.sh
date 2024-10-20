#!/bin/bash

# Array of models
models=("roberta" "deberta" "bart" "gpt2")


# Array of ranks
ranks=(4 8)

# Array to store all PIDs
declare -a PIDS=()

# Function to run the Python script with given parameters
run_model() {
    local model=$1
    local rank=$2
    local target_modules=$3
    local target_codes=$4
    local output_file="outputs/${model}_rank${rank}_${target_codes}.txt"
    
    echo "Running $model with rank $rank and target modules: $target_modules"
    # mkdir -p outputs
    python3 grm_class_lrft.py --model $model --rank $rank --target_modules "$target_modules" --target_codes $target_codes > "$output_file" 2>&1 &
    local pid=$!
    PIDS+=($pid)
    echo "Started process with PID: $pid"
    echo "Output will be saved to: $output_file"
    echo "----------------------------------------"
}

# Function to handle Ctrl+C
ctrl_c() {
    echo "Ctrl+C caught... Terminating all processes"
    for pid in "${PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            echo "Killing process $pid"
            kill $pid
        fi
    done
    exit 1
}

# Set up Ctrl+C trap
trap ctrl_c INT

# Loop through all combinations
for model in "${models[@]}"; do
    for rank in "${ranks[@]}"; do
        # Configuration 1: query, value
        # if [ "$model" == "roberta" ]; then
        #     run_model $model $rank "query value" "qv"
        # elif [ "$model" == "deberta" ]; then
        #     run_model $model $rank "query_proj value_proj" "qv"
        if [ "$model" == "bart" ]; then
            run_model $model $rank "q_proj v_proj" "qv"
        elif [ "$model" == "gpt2" ]; then
            run_model $model $rank "attn.c_attn" "qvk"
        fi

        # Configuration 2: query, value, key
        # if [ "$model" == "roberta" ]; then
        #     run_model $model $rank "query value key" "qvk"
        # elif [ "$model" == "deberta" ]; then
        #     run_model $model $rank "query_proj value_proj key_proj" "qvk"
        if [ "$model" == "bart" ]; then
            run_model $model $rank "q_proj v_proj k_proj" "qvk"
        elif [ "$model" == "gpt2" ]; then
            run_model $model $rank "attn.c_attn attn.c_proj" "qvko"
        fi
    done
done

echo "All processes have been started. PIDs: ${PIDS[@]}"
echo "Waiting for all processes to complete..."

# Wait for all background processes to finish
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "All models have been trained and evaluated!"
