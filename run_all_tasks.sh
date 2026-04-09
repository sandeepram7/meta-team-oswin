#!/bin/bash

# 1. Safely load .env variables
if [ -f .env ]; then
    echo "Loading variables from .env file..."
    set -a
    source .env
    set +a
fi

# 2. Ensure your key is set
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Please export your HF_TOKEN first or add it to your .env file!"
    exit 1
fi

echo "API Key found. Starting the Data Cleaning OpenEnv baseline evaluation..."

echo "=========================================="
echo "🚀 STARTING FULL 5-DATASET HACKATHON RUN"
echo "=========================================="

export MOCK=false

# 3. Loop through tasks 1 to 5 sequentially
for idx in {1..5}
do
    CURRENT_TASK="task_${idx}"
    # echo ""
    # echo "▶️ RUNNING: $CURRENT_TASK"
    # echo "------------------------------------------"
    
    # Pass the TASK_ID purely for this Python execution
    TASK_ID=$CURRENT_TASK python3 inference.py
    
    if [ $? -ne 0 ]; then
       # echo "⚠️ WARNING: $CURRENT_TASK encountered an error, but continuing to the next task..."
    fi
done

# echo ""
# echo "=========================================="
# echo "✅ FULL RUN COMPLETE."
# echo "=========================================="