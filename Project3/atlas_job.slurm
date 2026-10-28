#!/usr/bin/env bash
#SBATCH --job-name=heat_benchmark
#SBATCH --output=results/slurm_%j.out
#SBATCH --error=results/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=normal

# ==============================================================================
# SLURM Job Script for Atlas HPC Cluster (NUS)
# ==============================================================================
# To submit this job on Atlas:
#   sbatch atlas_job.slurm
#
# To check job status:
#   squeue -u $USER
# ==============================================================================

# 1. Load required modules on Atlas (uncomment/adjust according to Atlas module avail)
# module purge
# module load gcc
# module load openmpi

echo "Job started on: $(hostname) at $(date)"
echo "Running in directory: $(pwd)"

# 2. Run the automated experiment sweeper (5 repetitions per test)
# You can adjust N and STEPS as needed:
./sweep_experiments.sh 1000 1000

echo "Job finished at $(date)"

