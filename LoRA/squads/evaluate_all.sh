#!/bin/bash

# Array to store model paths
model_paths=(
    "saved_models/2.0/bart_lora_squad_4_qv/checkpoint-6180"
    "saved_models/2.0/bart_lora_squad_4_qvk/checkpoint-6180"
    "saved_models/2.0/bart_lora_squad_8_qv/checkpoint-6180"
    "saved_models/2.0/bart_lora_squad_8_qvk/checkpoint-6180"
    "saved_models/2.0/deberta_lora_squad2_4_qv/checkpoint-6165"
    "saved_models/2.0/deberta_lora_squad2_8_qv/checkpoint-6165"
    "saved_models/2.0/deberta_lora_squad2_4_qvk/checkpoint-6165"
    "saved_models/2.0/deberta_lora_squad2_8_qvk/checkpoint-6165"
    "saved_models/2.0/roberta_lora_squad_4_qv/checkpoint-6180"
    "saved_models/2.0/roberta_lora_squad_4_qvk/checkpoint-6180"
    "saved_models/2.0/roberta_lora_squad_8_qv/checkpoint-6180"
    "saved_models/2.0/roberta_lora_squad_8_qvk/checkpoint-6180"
)

# Function to run evaluation for a single model
run_evaluation() {
    local model_path=$1
    local squad_version=$2
    local model_name=$(basename $(dirname $model_path))
    local checkpoint=$(basename $model_path)
    local output_file="results/${squad_version}/${model_name}_${checkpoint}.txt"
    
    echo "Evaluating model: $model_path on SQuAD $squad_version"
    if [ "$squad_version" == "v1" ]; then
        python3 evaluate_squad.py "$model_path" > "$output_file" 2>&1
    else
        python3 evaluate_squad_v2.py "$model_path" > "$output_file" 2>&1
    fi
    echo "Evaluation complete for $model_path. Results saved in $output_file"
}

# Run evaluations in parallel for SQuAD v1.1
# echo "Starting parallel evaluations for SQuAD v1.1"
# for model_path in "${model_paths[@]}"; do
#     run_evaluation "$model_path" v1 &
# done

# Wait for all SQuAD v1.1 evaluations to complete
# wait

# Run evaluations in parallel for SQuAD v2.0
echo "Starting parallel evaluations for SQuAD v2.0"
for model_path in "${model_paths[@]}"; do
    run_evaluation "$model_path" v2 &
done

# Wait for all SQuAD v2.0 evaluations to complete
wait

echo "All evaluations complete!"
