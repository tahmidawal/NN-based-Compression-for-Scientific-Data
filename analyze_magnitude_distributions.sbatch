#!/bin/bash
#SBATCH --job-name=magnitude_analysis
#SBATCH --output=logs/magnitude_analysis_%j.out
#SBATCH --error=logs/magnitude_analysis_%j.err
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcqs-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting magnitude distribution analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Run the analysis
python analyze_magnitude_distributions.py

echo "Analysis completed!"
echo "Results saved in: magnitude_distributions/"