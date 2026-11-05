#!/usr/bin/env bash
# ==============================================================================
# Automated Scaling Experiment Sweeper (for Atlas HPC & Local Benchmarking)
# ==============================================================================
# Benchmarks both serial variants and the MPI configurations of the 2D Heat Equation:
#   1. Original unoptimized serial reference
#   2. Optimized serial baseline (used for MPI speedup and efficiency)
#   3. 1D Row Decomposition + Blocking Communication (heat_mpi_implicit.cpp)
#   4. 1D Row Decomposition + Non-blocking Overlap (heat_mpi_nonblocking_implicit.cpp)
#   5. 2D Block Decomposition + Blocking Communication (heat_mpi_block_implicit.cpp)
#   6. 2D Block Decomposition + Non-blocking Overlap (heat_mpi_block_nonblocking_implicit.cpp)
#
# One row per repetition is written to ./results/all_results.csv. MPI outputs
# include average-rank and maximum-rank compute/communication times.
# ==============================================================================

set -euo pipefail

# Simulation parameters (Default: N=3000, STEPS=1000, RUNS=5)
N="${1:-3000}"
STEPS="${2:-1000}"
RUNS="${3:-5}"

# Target process counts for the requested scaling study.
# PBS filters this list against the allocated MPI slots below.
DEFAULT_PROCS="1 2 4 8 16 32"

if [ -n "${PBS_NODEFILE:-}" ] && [ -z "${PROCS_LIST:-}" ]; then
  TOTAL_SLOTS=$(wc -l < "$PBS_NODEFILE")
  echo "PBS environment detected: $TOTAL_SLOTS slots allocated."
  VALID_PROCS=""
  for p in $DEFAULT_PROCS; do
    if [ "$p" -le "$TOTAL_SLOTS" ]; then
      VALID_PROCS="$VALID_PROCS $p"
    fi
  done
  PROCS_LIST="${VALID_PROCS# }"
else
  PROCS_LIST="${PROCS_LIST:-$DEFAULT_PROCS}"
fi

echo "=========================================================="
echo " Starting Full HPC Experiment Benchmark Sweep"
echo " Problem Size: N=$N, STEPS=$STEPS, Repetitions=$RUNS"
echo " Process Counts to Sweep: $PROCS_LIST"
echo "=========================================================="

# 1. Original unoptimized serial reference. This does not set the efficiency baseline.
echo -e "\n>>> [1/6] Benchmarking Configuration 1: Unoptimized Serial Reference..."
./run_scaling_benchmark.sh heat_serial_implicit_unoptimized.cpp 1 "$RUNS" "$N" "$STEPS"

# 2. Optimized serial baseline. MPI speedup and efficiency use this timing.
echo -e "\n>>> [2/6] Benchmarking Configuration 2: Optimized Serial Baseline..."
./run_scaling_benchmark.sh heat_serial_implicit_optimized.cpp 1 "$RUNS" "$N" "$STEPS"

# 3. MPI 1D Row Decomposition + Blocking Communication
echo -e "\n>>> [3/6] Benchmarking Configuration 3: 1D Row Decomposition (Blocking)..."
for p in $PROCS_LIST; do
  ./run_scaling_benchmark.sh heat_mpi_implicit.cpp "$p" "$RUNS" "$N" "$STEPS"
done

# 4. MPI 1D Row Decomposition + Non-blocking Communication
echo -e "\n>>> [4/6] Benchmarking Configuration 4: 1D Row Decomposition (Non-blocking Overlap)..."
for p in $PROCS_LIST; do
  ./run_scaling_benchmark.sh heat_mpi_nonblocking_implicit.cpp "$p" "$RUNS" "$N" "$STEPS"
done

# 5. MPI 2D Block Decomposition + Blocking Communication
echo -e "\n>>> [5/6] Benchmarking Configuration 5: 2D Block Decomposition (Blocking)..."
for p in $PROCS_LIST; do
  ./run_scaling_benchmark.sh heat_mpi_block_implicit.cpp "$p" "$RUNS" "$N" "$STEPS"
done

# 6. MPI 2D Block Decomposition + Non-blocking Communication
echo -e "\n>>> [6/6] Benchmarking Configuration 6: 2D Block Decomposition (Non-blocking Overlap)..."
for p in $PROCS_LIST; do
  ./run_scaling_benchmark.sh heat_mpi_block_nonblocking_implicit.cpp "$p" "$RUNS" "$N" "$STEPS"
done

echo -e "\n=========================================================="
echo " Benchmark Sweep Finished Successfully!"
echo " Results Directory : ./results/"
echo " Summary CSV Table : ./results/all_results.csv"
echo "=========================================================="
