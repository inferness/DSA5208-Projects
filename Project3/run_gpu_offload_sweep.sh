#!/usr/bin/env bash
# Run the 2 x 3 OpenMP GPU mapping/thread-limit experiment matrix.
# Usage: ./run_gpu_offload_sweep.sh [runs=5] [N=3000] [steps=1000] [lambda=0.05] [max_iters=25] [tol=1e-5]

set -euo pipefail

RUNS="${1:-5}"
N="${2:-3000}"
STEPS="${3:-1000}"
LAMBDA="${4:-0.05}"
MAX_ITERS="${5:-25}"
TOL="${6:-1e-5}"
GPU_CXX="${GPU_CXX:-nvc++}"
CPU_CXX="${CPU_CXX:-g++}"
GPU_CXXFLAGS="${GPU_CXXFLAGS:--O3 -std=c++17 -mp=gpu -Minfo=mp}"
GPU_RESULTS_ROOT="${GPU_RESULTS_DIR:-results/openmp_gpu_sweep}"
BIN_DIR="${BIN_DIR:-bin}"
THREAD_LIMITS=(128 256 512)
MAPPINGS=(collapse2 row_only)

if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: runs must be a positive integer." >&2
  exit 2
fi
if ! command -v "$GPU_CXX" >/dev/null 2>&1; then
  echo "Error: GPU compiler '$GPU_CXX' not found. Load Vanda's NVIDIA HPC SDK module." >&2
  exit 2
fi
if ! command -v "$CPU_CXX" >/dev/null 2>&1; then
  echo "Error: CPU reference compiler '$CPU_CXX' not found." >&2
  exit 2
fi

mkdir -p "$BIN_DIR" "$GPU_RESULTS_ROOT"
read -r -a GPU_FLAGS <<< "$GPU_CXXFLAGS"
unset OMP_NUM_TEAMS
export OMP_TARGET_OFFLOAD=MANDATORY

{
  echo "OpenMP GPU sweep started: $(date)"
  echo "Node: $(hostname)"
  echo "GPU compiler: $GPU_CXX"
  "$GPU_CXX" --version 2>&1 || true
  echo "CPU reference compiler: $CPU_CXX"
  "$CPU_CXX" --version 2>&1 | head -n 2 || true
  echo "GPU inventory:"
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L 2>&1 || true; else echo "nvidia-smi not found"; fi
  echo "Loaded modules:"
  module list 2>&1 || true
  echo "Grid/steps: N=$N STEPS=$STEPS; runs/configuration=$RUNS"
  echo "OMP_NUM_TEAMS: automatic runtime default"
  echo "OMP_TEAMS_THREAD_LIMIT values: ${THREAD_LIMITS[*]}"
} | tee "$GPU_RESULTS_ROOT/gpu_offload_sweep_toolchain.txt"

echo "[1/4] Compiling optimized CPU reference..."
"$CPU_CXX" -O3 -std=c++17 heat_serial_implicit_optimized.cpp \
  -o "$BIN_DIR/heat_serial_implicit_optimized"
CPU_REFERENCE_LOG="$GPU_RESULTS_ROOT/heat_serial_implicit_optimized_reference.log"
OMP_NUM_THREADS=1 "$BIN_DIR/heat_serial_implicit_optimized" \
  "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" > "$CPU_REFERENCE_LOG" 2>&1
echo "      CPU reference saved in $CPU_REFERENCE_LOG"

echo "[2/4] Compiling collapse(2) GPU variant..."
"$GPU_CXX" "${GPU_FLAGS[@]}" heat_openmp_gpu_offload.cpp \
  -o "$BIN_DIR/heat_openmp_gpu_collapse2" 2>&1 | tee "$GPU_RESULTS_ROOT/compile_collapse2.log"

echo "[3/4] Compiling row-only GPU variant..."
"$GPU_CXX" "${GPU_FLAGS[@]}" -DGPU_ROW_ONLY=1 heat_openmp_gpu_offload.cpp \
  -o "$BIN_DIR/heat_openmp_gpu_row_only" 2>&1 | tee "$GPU_RESULTS_ROOT/compile_row_only.log"

CSV_FILE="$GPU_RESULTS_ROOT/all_gpu_sweep_results.csv"
echo "thread_limit,mapping,num_teams,runs,mean_time_s,min_time_s,max_time_s,stddev_s,mean_jacobi_iters,max_cpu_diff" > "$CSV_FILE"

for THREAD_LIMIT in "${THREAD_LIMITS[@]}"; do
  for MAPPING in "${MAPPINGS[@]}"; do
    CONFIG_DIR="$GPU_RESULTS_ROOT/thread_limit_${THREAD_LIMIT}/${MAPPING}"
    mkdir -p "$CONFIG_DIR/logs"
    if [ "$MAPPING" = "collapse2" ]; then
      BINARY="$BIN_DIR/heat_openmp_gpu_collapse2"
    else
      BINARY="$BIN_DIR/heat_openmp_gpu_row_only"
    fi

    echo "=========================================================="
    echo "Configuration: mapping=$MAPPING OMP_TEAMS_THREAD_LIMIT=$THREAD_LIMIT"
    RUN_TIMES=()
    RUN_ITERS=()
    RUN_DIFFS=()

    for ((run = 1; run <= RUNS; run++)); do
      LOG_FILE="$CONFIG_DIR/logs/run${run}_$(date +'%Y%m%d_%H%M%S').log"
      echo "Run $run / $RUNS ..."
      OMP_TEAMS_THREAD_LIMIT="$THREAD_LIMIT" \
        OMP_TARGET_OFFLOAD=MANDATORY GPU_RESULTS_DIR="$CONFIG_DIR" \
        "$BINARY" "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" \
        > "$LOG_FILE" 2>&1

      if ! grep -q "Selected target.*GPU confirmed" "$LOG_FILE"; then
        cat "$LOG_FILE"
        echo "Error: GPU execution was not confirmed in $LOG_FILE" >&2
        exit 1
      fi
      if ! grep -q "CPU baseline check.*PASSED" "$LOG_FILE"; then
        cat "$LOG_FILE"
        echo "Error: CPU validation did not pass in $LOG_FILE" >&2
        exit 1
      fi

      cat "$LOG_FILE"
      TIME_VAL="$(awk -F: '/^Execution time/ {gsub(/ seconds/, "", $2); gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
      ITER_VAL="$(awk -F: '/^Total Jacobi iters/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
      DIFF_VAL="$(awk -F: '/^Max difference vs CPU baseline/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
      if [ -z "$TIME_VAL" ] || [ -z "$ITER_VAL" ] || [ -z "$DIFF_VAL" ]; then
        echo "Error: missing time, iteration count, or CPU difference in $LOG_FILE" >&2
        exit 1
      fi
      RUN_TIMES+=("$TIME_VAL")
      RUN_ITERS+=("$ITER_VAL")
      RUN_DIFFS+=("$DIFF_VAL")
    done

    STATS="$(awk -v n="${#RUN_TIMES[@]}" '{ v[NR]=$1; sum+=$1 }
      END {
        mean=sum/n; min=v[1]; max=v[1]; ss=0;
        for (i=1; i<=n; i++) {
          if (v[i]<min) min=v[i];
          if (v[i]>max) max=v[i];
          d=v[i]-mean; ss+=d*d;
        }
        sd=(n>1) ? sqrt(ss/(n-1)) : 0;
        printf "%.6f %.6f %.6f %.6f\n", mean,min,max,sd;
      }' < <(printf '%s\n' "${RUN_TIMES[@]}") )"
    read -r MEAN_TIME MIN_TIME MAX_TIME STDDEV_TIME <<< "$STATS"
    MEAN_ITERS="$(awk '{sum+=$1} END {printf "%.3f", sum/NR}' < <(printf '%s\n' "${RUN_ITERS[@]}"))"
    MAX_DIFF="$(awk 'NR==1 || $1>max {max=$1} END {printf "%.9e", max}' < <(printf '%s\n' "${RUN_DIFFS[@]}"))"

    {
      echo "OpenMP GPU sweep configuration"
      echo "Mapping: $MAPPING"
      echo "OMP_TEAMS_THREAD_LIMIT: $THREAD_LIMIT"
      echo "OMP_NUM_TEAMS: automatic runtime default"
      echo "Runs: $RUNS"
      echo "Mean time: $MEAN_TIME s"
      echo "Min time: $MIN_TIME s"
      echo "Max time: $MAX_TIME s"
      echo "Sample stddev: $STDDEV_TIME s"
      echo "Mean Jacobi iterations: $MEAN_ITERS"
      echo "Max CPU difference: $MAX_DIFF"
      echo "Logs: $CONFIG_DIR/logs/"
    } | tee "$CONFIG_DIR/summary.txt"

    echo "$THREAD_LIMIT,$MAPPING,automatic,$RUNS,$MEAN_TIME,$MIN_TIME,$MAX_TIME,$STDDEV_TIME,$MEAN_ITERS,$MAX_DIFF" >> "$CSV_FILE"
  done
done

echo "=========================================================="
echo "GPU mapping/thread-limit sweep complete."
echo "CSV: $CSV_FILE"
echo "Per-configuration output: $GPU_RESULTS_ROOT/thread_limit_<N>/<mapping>/"
