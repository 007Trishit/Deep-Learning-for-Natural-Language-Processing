#!/bin/bash

# Array to store child PIDs
pids=()

# Function to kill all child processes
cleanup() {
    echo "Interruption received. Terminating all processes..."
    for pid in "${pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null
    done
    wait
    echo "All processes terminated."
    exit 1
}

# Set up trap to call cleanup function on Ctrl+C (SIGINT)
trap cleanup SIGINT

# Check if we have 4 arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <script1> <script2> <script3>"
    exit 1
fi

# Setup Conda
eval "$(conda shell.bash hook)"
conda activate dl-nlp

# Function to run a single finetuning script
run_finetuning() {
    echo "Starting: $1"
    python3 "$1" > "outputs/${1%.*}_output7.log" 2>&1 &
    pid=$!
    pids+=($pid)
    echo "PID: $pid"
}

# Run 4 finetuning scripts in parallel
run_finetuning "$1"
run_finetuning "$2"
run_finetuning "$3"
# run_finetuning "$4"

# Wait for all background jobs to finish
wait

echo "All finetuning jobs completed."
