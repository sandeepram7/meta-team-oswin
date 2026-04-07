#!/bin/bash

# Ensure your key is set (it should be if you've been running inference)
if [ -z "$GROQ_API_KEY" ]; then
    echo "ERROR: Please export your GROQ_API_KEY first!"
    exit 1
fi

echo "=========================================="
echo "🚀 STARTING FULL 5-DATASET HACKATHON RUN"
echo "=========================================="

export MOCK=false

for idx in {1..5}
do
    export TASK_ID="task_${idx}"
    echo ""
    echo "▶️ RUNNING: $TASK_ID"
    echo "------------------------------------------"
    python inference.py
done

echo ""
echo "=========================================="
echo "✅ FULL RUN COMPLETE."
echo "=========================================="
