#!/usr/bin/env bash
# ==============================================================================
# Automated Scaling Experiment Sweeper (for Atlas / HPC)
# ==============================================================================
# Automatically compiles and benchmarks:
#   1. Serial Baseline (5 runs)
#   2. MPI 1D Row Decomposition across processes: 1, 2, 4, 8, 16 (5 runs each)
#   3. MPI 2D Block Decomposition across processes: 1, 2, 4, 8, 16 (5 runs each)
#   4. Hybrid MPI+OpenMP across combinations (e.g. 2x4, 4x2)
#
# All results, statistics (Mean, Min, Max, StdDev), and raw logs are written
# to ./results/ and appended to ./results/all_results.csv.
# ==============================================================================

set -euo pipefail

RUNS=5

# Grid parameters (adjust as requested by your assignment):
# Default: N=1000, STEPS=1000
N="${1:-1000}"
STEPS="${2:-1000}"

echo "=========================================================="
echo " Starting Full HPC Experiment Benchmark Sweep"
echo " Problem Size: N=$N, STEPS=$STEPS, Repetitions=$RUNS"
echo "=========================================================="

# 1. Serial Implicit Baseline
echo -e "\n>>> [1/4] Benchmarking Serial Baseline..."
./run_benchmark.sh heat_serial_implicit.cpp 1 1 "$RUNS" "$N" "$STEPS"

# 2. MPI 1D Implicit Scaling Sweep
echo -e "\n>>> [2/4] Benchmarking MPI 1D Decomposition (p = 1, 2, 4, 8, 16)..."
for p in 1 2 4 8 16; do
  ./run_benchmark.sh heat_mpi_implicit.cpp "$p" 1 "$RUNS" "$N" "$STEPS"
done

# 3. MPI 2D Block Implicit Scaling Sweep (if file exists)
if [ -f "heat_mpi_block_implicit.cpp" ]; then
  echo -e "\n>>> [3/4] Benchmarking MPI 2D Block Decomposition (p = 1, 2, 4, 8, 16)..."
  for p in 1 2 4 8 16; do
    ./run_benchmark.sh heat_mpi_block_implicit.cpp "$p" 1 "$RUNS" "$N" "$STEPS"
  done
fi

# 4. Hybrid MPI + OpenMP Implicit Sweep (if file exists)
if [ -f "heat_mpi_openMP_implicit.cpp" ]; then
  echo -e "\n>>> [4/4] Benchmarking Hybrid MPI + OpenMP..."
  ./run_benchmark.sh heat_mpi_openMP_implicit.cpp 1 4 "$RUNS" "$N" "$STEPS"
  ./run_benchmark.sh heat_mpi_openMP_implicit.cpp 2 4 "$RUNS" "$N" "$STEPS"
  ./run_benchmark.sh heat_mpi_openMP_implicit.cpp 4 2 "$RUNS" "$N" "$STEPS"
  ./run_benchmark.sh heat_mpi_openMP_implicit.cpp 4 4 "$RUNS" "$N" "$STEPS"
fi

echo -e "\n=========================================================="
echo " Benchmark Sweep Finished!"
echo " Results Directory: ./results/"
echo " Summary CSV Table : ./results/all_results.csv"
echo "=========================================================="

