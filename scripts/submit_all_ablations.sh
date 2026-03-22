#!/bin/bash
# Batch submission script for all ablation experiments
# Usage: bash submit_all_ablations.sh

echo "=========================================="
echo "Submitting all ablation experiments"
echo "=========================================="

cd "$(dirname "$0")"

experiments=(
    "submit_full_model.sh"
    "submit_no_texture.sh"
    "submit_no_memory.sh"
    "submit_basic_memory.sh"
    "submit_no_aux_losses.sh"
    "submit_vanilla_var.sh"
)

for script in "${experiments[@]}"; do
    if [ -f "$script" ]; then
        echo ""
        echo "Submitting: $script"
        sbatch "$script"
        if [ $? -eq 0 ]; then
            echo "✓ Successfully submitted $script"
        else
            echo "✗ Failed to submit $script"
        fi
        sleep 2  # Avoid overwhelming the scheduler
    else
        echo "✗ Script not found: $script"
    fi
done

echo ""
echo "=========================================="
echo "All submissions completed"
echo "Check job status with: squeue -u $USER"
echo "=========================================="
