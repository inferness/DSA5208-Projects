#!/usr/bin/env bash
# Sweep six hybrid MPI/OpenMP layouts on one 32-core node.
# Usage: ./run_hybrid_openmp_sweep.sh [runs=5] [N=3000] [steps=1000]

set -euo pipefail

RUNS="${1:-5}"
N="${2:-3000}"
STEPS="${3:-1000}"
LAMBDA="${4:-0.05}"
MAX_ITERS="${5:-25}"
TOL="${6:-1e-5}"
MPI_CXX="${MPI_CXX:-mpicxx}"
MPI_RUN="${MPI_RUN:-mpirun}"
RESULTS_ROOT="${HYBRID_RESULTS_DIR:-results/hybrid_openmp}"
BIN_DIR="${BIN_DIR:-bin}"
PROCS=(1 2 4 8 16 32)
THREADS=(32 16 8 4 2 1)

if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: runs must be a positive integer." >&2
  exit 2
fi
if ! command -v "$MPI_CXX" >/dev/null 2>&1; then
  echo "Error: MPI compiler '$MPI_CXX' not found." >&2
  exit 2
fi
if ! command -v "$MPI_RUN" >/dev/null 2>&1; then
  echo "Error: MPI launcher '$MPI_RUN' not found." >&2
  exit 2
fi

mkdir -p "$BIN_DIR" "$RESULTS_ROOT"
export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close

{
  echo "Hybrid MPI/OpenMP sweep started: $(date)"
  echo "Node: $(hostname)"
  echo "MPI compiler: $MPI_CXX"
  "$MPI_CXX" --version 2>&1 | head -n 2 || true
  echo "MPI launcher: $MPI_RUN"
  "$MPI_RUN" --version 2>&1 | head -n 3 || true
  echo "Grid/steps: N=$N STEPS=$STEPS; repetitions/configuration=$RUNS"
  echo "Layouts: 1x32, 2x16, 4x8, 8x4, 16x2, 32x1 (32 workers each)"
  echo "OpenMP placement: OMP_PLACES=$OMP_PLACES OMP_PROC_BIND=$OMP_PROC_BIND"
  echo "PBS node file: ${PBS_NODEFILE:-not set}"
} | tee "$RESULTS_ROOT/toolchain.txt"

echo "[1/3] Compiling the optimized serial reference..."
"$MPI_CXX" -O3 -std=c++17 heat_serial_implicit_optimized.cpp \
  -o "$BIN_DIR/heat_serial_implicit_optimized"
OMP_NUM_THREADS=1 "$BIN_DIR/heat_serial_implicit_optimized" \
  "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" \
  > "$RESULTS_ROOT/serial_reference.log" 2>&1

echo "[2/3] Compiling row-decomposed nonblocking hybrid MPI+OpenMP..."
"$MPI_CXX" -O3 -std=c++17 -fopenmp heat_mpi_openMP_implicit.cpp \
  -o "$BIN_DIR/heat_mpi_openMP_implicit" 2>&1 | tee "$RESULTS_ROOT/compile.log"

CSV_FILE="$RESULTS_ROOT/all_hybrid_results.csv"
echo "mpi_ranks,threads_per_rank,requested_total_workers,runs,mean_time_s,min_time_s,max_time_s,stddev_s,observed_threads_per_rank,mean_jacobi_iters,max_residual,max_baseline_diff" > "$CSV_FILE"

for i in "${!PROCS[@]}"; do
  P="${PROCS[$i]}"
  T="${THREADS[$i]}"
  CONFIG_DIR="$RESULTS_ROOT/ranks_${P}_threads_${T}"
  mkdir -p "$CONFIG_DIR/logs"
  RUN_TIMES=()
  RUN_THREADS=()
  RUN_ITERS=()
  RUN_RESIDUALS=()
  RUN_DIFFS=()

  echo "=========================================================="
  echo "Configuration: MPI ranks=$P OpenMP threads/rank=$T (32 workers)"
  for ((run = 1; run <= RUNS; run++)); do
    LOG_FILE="$CONFIG_DIR/logs/run${run}_$(date +'%Y%m%d_%H%M%S').log"
    echo "Run $run / $RUNS ..."
    OMP_NUM_THREADS="$T" "$MPI_RUN" -np "$P" \
      --map-by "ppr:${P}:node:PE=${T}" --bind-to core \
      "$BIN_DIR/heat_mpi_openMP_implicit" "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" \
      > "$LOG_FILE" 2>&1

    if ! grep -q "Execution time" "$LOG_FILE"; then
      cat "$LOG_FILE"
      echo "Error: MPI run did not produce timing output; see $LOG_FILE" >&2
      exit 1
    fi
    TIME_VAL="$(awk -F: '/^Execution time/ {gsub(/ seconds/, "", $2); gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
    THREAD_VAL="$(awk -F: '/^OpenMP threads\/proc/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
    ITER_VAL="$(awk -F: '/^Total Jacobi iters/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
    RESIDUAL_VAL="$(awk -F: '/^Max absolute residual/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
    DIFF_VAL="$(awk -F: '/Max Absolute Diff/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
    if [ -z "$TIME_VAL" ] || [ -z "$THREAD_VAL" ] || [ -z "$ITER_VAL" ]; then
      cat "$LOG_FILE"
      echo "Error: timing or runtime configuration details missing in $LOG_FILE" >&2
      exit 1
    fi
    echo "Runtime reports $THREAD_VAL threads per rank."
    RUN_TIMES+=("$TIME_VAL")
    RUN_THREADS+=("$THREAD_VAL")
    RUN_ITERS+=("$ITER_VAL")
    RUN_RESIDUALS+=("${RESIDUAL_VAL:-0}")
    RUN_DIFFS+=("${DIFF_VAL:-NA}")
  done

  STATS="$(awk -v n="${#RUN_TIMES[@]}" '{ v[NR]=$1; sum+=$1 }
    END { mean=sum/n; min=v[1]; max=v[1]; ss=0;
      for (i=1; i<=n; i++) { if (v[i]<min) min=v[i]; if (v[i]>max) max=v[i]; d=v[i]-mean; ss+=d*d }
      sd=(n>1) ? sqrt(ss/(n-1)) : 0; printf "%.6f %.6f %.6f %.6f\n", mean,min,max,sd;
    }' < <(printf '%s\n' "${RUN_TIMES[@]}") )"
  read -r MEAN_TIME MIN_TIME MAX_TIME STDDEV_TIME <<< "$STATS"
  OBS_THREADS="$(awk '{sum+=$1} END {printf "%.2f", sum/NR}' < <(printf '%s\n' "${RUN_THREADS[@]}"))"
  MEAN_ITERS="$(awk '{sum+=$1} END {printf "%.3f", sum/NR}' < <(printf '%s\n' "${RUN_ITERS[@]}"))"
  MAX_RESIDUAL="$(awk 'BEGIN{max=0} {v=$1+0; if (v>max) max=v} END {printf "%.9e", max}' < <(printf '%s\n' "${RUN_RESIDUALS[@]}"))"
  MAX_DIFF="$(awk '$1!="NA" {v=$1+0; if (!found || v>max) max=v; found=1} END {if(found) printf "%.9e", max; else print "NA"}' < <(printf '%s\n' "${RUN_DIFFS[@]}"))"

  {
    echo "MPI ranks: $P"
    echo "Requested OpenMP threads per rank: $T"
    echo "Requested total workers: $((P * T))"
    echo "Mean observed OpenMP threads per rank: $OBS_THREADS"
    echo "Runs: $RUNS"
    echo "Mean time: $MEAN_TIME s"
    echo "Min time: $MIN_TIME s"
    echo "Max time: $MAX_TIME s"
    echo "Sample stddev: $STDDEV_TIME s"
    echo "Mean Jacobi iterations: $MEAN_ITERS"
    echo "Maximum residual: $MAX_RESIDUAL"
    echo "Maximum baseline difference: $MAX_DIFF"
    echo "Logs: $CONFIG_DIR/logs/"
  } | tee "$CONFIG_DIR/summary.txt"

  echo "$P,$T,$((P * T)),$RUNS,$MEAN_TIME,$MIN_TIME,$MAX_TIME,$STDDEV_TIME,$OBS_THREADS,$MEAN_ITERS,$MAX_RESIDUAL,$MAX_DIFF" >> "$CSV_FILE"
done

echo "=========================================================="
echo "Hybrid MPI/OpenMP sweep complete."
echo "CSV: $CSV_FILE"
