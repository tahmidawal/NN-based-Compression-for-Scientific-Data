#!/bin/bash

# Master Script: Submit Parallel Architecture Search Jobs
# This script submits 5 parallel jobs, one for each latent dimension

echo "=========================================="
echo "SWAE Parallel Architecture Search"
echo "=========================================="
echo "Submitting 5 parallel jobs for latent dimensions: 4, 8, 16, 32, 64"
echo "Each job will test 32 configurations (4 architectures × 2 LRs × 2 batch sizes × 2 lambdas)"
echo "Total configurations across all jobs: 160"
echo "Expected runtime: ~8 hours per job (parallel execution)"
echo "=========================================="

# Check if logs directory exists
mkdir -p logs

# Store job IDs for tracking
JOB_IDS=()

# Submit latent dimension 4 job
echo "🚀 Submitting job for latent dimension 4 (85.8:1 compression)..."
job_id_4=$(sbatch train_swae_arch_search_latent4.sbatch | grep -o '[0-9]*')
JOB_IDS+=($job_id_4)
echo "   Job ID: $job_id_4"

# Submit latent dimension 8 job
echo "🚀 Submitting job for latent dimension 8 (42.9:1 compression)..."
job_id_8=$(sbatch train_swae_arch_search_latent8.sbatch | grep -o '[0-9]*')
JOB_IDS+=($job_id_8)
echo "   Job ID: $job_id_8"

# Submit latent dimension 16 job
echo "🚀 Submitting job for latent dimension 16 (21.4:1 compression)..."
job_id_16=$(sbatch train_swae_arch_search_latent16.sbatch | grep -o '[0-9]*')
JOB_IDS+=($job_id_16)
echo "   Job ID: $job_id_16"

# Submit latent dimension 32 job
echo "🚀 Submitting job for latent dimension 32 (10.7:1 compression)..."
job_id_32=$(sbatch train_swae_arch_search_latent32.sbatch | grep -o '[0-9]*')
JOB_IDS+=($job_id_32)
echo "   Job ID: $job_id_32"

# Submit latent dimension 64 job
echo "🚀 Submitting job for latent dimension 64 (5.4:1 compression)..."
job_id_64=$(sbatch train_swae_arch_search_latent64.sbatch | grep -o '[0-9]*')
JOB_IDS+=($job_id_64)
echo "   Job ID: $job_id_64"

echo ""
echo "=========================================="
echo "All Jobs Submitted Successfully!"
echo "=========================================="
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "📊 Monitor progress with:"
echo "   squeue -u \$USER"
echo "   sacct -j ${JOB_IDS[0]},${JOB_IDS[1]},${JOB_IDS[2]},${JOB_IDS[3]},${JOB_IDS[4]} --format=JobID,JobName,State,Start,End,Elapsed"
echo ""
echo "📁 Check logs in real-time:"
echo "   tail -f logs/swae_arch_latent4_${job_id_4}.out"
echo "   tail -f logs/swae_arch_latent8_${job_id_8}.out"
echo "   tail -f logs/swae_arch_latent16_${job_id_16}.out"
echo "   tail -f logs/swae_arch_latent32_${job_id_32}.out"
echo "   tail -f logs/swae_arch_latent64_${job_id_64}.out"
echo ""
echo "🔗 After completion, consolidate results with:"
echo "   ./consolidate_parallel_results.sh"
echo ""
echo "⏱️  Expected completion time: ~8 hours"
echo "🎯 Total configurations to test: 160 (32 per latent dimension)"

# Save job information for later consolidation
echo "${JOB_IDS[@]}" > .parallel_job_ids.txt
echo "Job IDs saved to .parallel_job_ids.txt for result consolidation"

echo ""
echo "🎉 All parallel architecture search jobs are now running!" 