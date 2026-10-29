#!/usr/bin/env bash
# ==============================================================================
# Automated Scaling Experiment Sweeper (for Atlas HPC & Local Benchmarking)
# ==============================================================================
# Benchmarks all 5 configurations of the 2D Heat Equation Implicit Solver:
#   1. Serial Baseline (heat_serial_implicit.cpp)
#   2. 1D Row Decomposition + Blocking Communication (heat_mpi_implicit.cpp)
#   3. 1D Row Decomposition + Non-blocking Overlap (heat_mpi_nonblocking_implicit.cpp)
#   4. 2D Block Decomposition + Blocking Communication (heat_mpi_block_implicit.cpp)
#   5. 2D Block Decomposition + Non-blocking Overlap (heat_mpi_block_nonblocking_implicit.cpp)
#   [Bonus: Hybrid MPI + OpenMP (heat_mpi_openMP_implicit.cpp)]
#
# All execution times, speedups, and statistics (Mean, Min, Max, StdDev) are
# automatically computed and appended to ./results/all_results.csv.
# ==============================================================================

set -euo pipefail

# Simulation parameters (Default: N=3000, STEPS=1000, RUNS=5)
N="${1:-3000}"
STEPS="${2:-1000}"
RUNS="${3:-5}"

# Target process counts: 1, 4, 16, 32
# (1, 4, 16 are square 2D grids 1x1, 2x2, 4x4; 32 utilizes the full 32-core allocation)
DEFAULT_PROCS="1 4 16 32"

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

# 1. Serial Implicit Baseline
echo -e "\n>>> [1/5] Benchmarking Configuration 1: Serial Baseline..."
./run_benchmark.sh heat_serial_implicit.cpp 1 1 "$RUNS" "$N" "$STEPS"

# 2. MPI 1D Row Decomposition + Blocking Communication
echo -e "\n>>> [2/5] Benchmarking Configuration 2: 1D Row Decomposition (Blocking)..."
for p in $PROCS_LIST; do
  ./run_benchmark.sh heat_mpi_implicit.cpp "$p" 1 "$RUNS" "$N" "$STEPS"
done

# 3. MPI 1D Row Decomposition + Non-blocking Communication
echo -e "\n>>> [3/5] Benchmarking Configuration 3: 1D Row Decomposition (Non-blocking Overlap)..."
for p in $PROCS_LIST; do
  ./run_benchmark.sh heat_mpi_nonblocking_implicit.cpp "$p" 1 "$RUNS" "$N" "$STEPS"
done

# 4. MPI 2D Block Decomposition + Blocking Communication
echo -e "\n>>> [4/5] Benchmarking Configuration 4: 2D Block Decomposition (Blocking)..."
for p in $PROCS_LIST; do
  ./run_benchmark.sh heat_mpi_block_implicit.cpp "$p" 1 "$RUNS" "$N" "$STEPS"
done

# 5. MPI 2D Block Decomposition + Non-blocking Communication
echo -e "\n>>> [5/5] Benchmarking Configuration 5: 2D Block Decomposition (Non-blocking Overlap)..."
for p in $PROCS_LIST; do
  ./run_benchmark.sh heat_mpi_block_nonblocking_implicit.cpp "$p" 1 "$RUNS" "$N" "$STEPS"
done

# Optional: Hybrid MPI + OpenMP (if present)
if [ -f "heat_mpi_openMP_implicit.cpp" ] && [ "${RUN_HYBRID:-false}" = "true" ]; then
  echo -e "\n>>> [Bonus] Benchmarking Hybrid MPI + OpenMP..."
  ./run_benchmark.sh heat_mpi_openMP_implicit.cpp 1 4 "$RUNS" "$N" "$STEPS"
  ./run_benchmark.sh heat_mpi_openMP_implicit.cpp 2 4 "$RUNS" "$N" "$STEPS"
  ./run_benchmark.sh heat_mpi_openMP_implicit.cpp 4 2 "$RUNS" "$N" "$STEPS"
  ./run_benchmark.sh heat_mpi_openMP_implicit.cpp 4 4 "$RUNS" "$N" "$STEPS"
fi

echo -e "\n=========================================================="
echo " Benchmark Sweep Finished Successfully!"
echo " Results Directory : ./results/"
echo " Summary CSV Table : ./results/all_results.csv"
echo "=========================================================="
