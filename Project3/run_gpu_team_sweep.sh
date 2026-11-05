#!/usr/bin/env bash
# Sweep requested OpenMP team counts with a fixed 256-thread team limit.
# Usage: ./run_gpu_team_sweep.sh [runs=5] [N=3000] [steps=1000] [lambda=0.05] [max_iters=25] [tol=1e-5]

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
GPU_RESULTS_ROOT="${GPU_RESULTS_DIR:-gpu_team_sweep_results}"
TEAM_COUNTS=(256 512 1024 2048)
THREAD_LIMIT=256

if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: runs must be a positive integer." >&2
  exit 2
fi
for compiler in "$GPU_CXX" "$CPU_CXX"; do
  if ! command -v "$compiler" >/dev/null 2>&1; then
    echo "Error: compiler '$compiler' not found." >&2
    exit 2
  fi
done

mkdir -p bin results/logs "$GPU_RESULTS_ROOT"
read -r -a GPU_FLAGS <<< "$GPU_CXXFLAGS"
export OMP_TARGET_OFFLOAD=MANDATORY
export OMP_TEAMS_THREAD_LIMIT="$THREAD_LIMIT"

{
  echo "OpenMP GPU team-count sweep started: $(date)"
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
  echo "Loop mapping: collapse(2)"
  echo "OMP_TEAMS_THREAD_LIMIT: $THREAD_LIMIT"
  echo "Requested team counts: ${TEAM_COUNTS[*]}"
} | tee "$GPU_RESULTS_ROOT/gpu_team_sweep_toolchain.txt"

echo "[1/3] Compiling optimized CPU reference..."
"$CPU_CXX" -O3 -std=c++17 heat_serial_implicit_optimized.cpp -o bin/heat_serial_implicit_optimized
CPU_REFERENCE_LOG="$GPU_RESULTS_ROOT/heat_serial_implicit_optimized_reference.log"
OMP_NUM_THREADS=1 ./bin/heat_serial_implicit_optimized \
  "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" > "$CPU_REFERENCE_LOG" 2>&1

echo "[2/3] Compiling collapse(2) GPU code with actual-team reporting..."
"$GPU_CXX" "${GPU_FLAGS[@]}" heat_openmp_gpu_offload.cpp \
  -o bin/heat_openmp_gpu_team_sweep 2>&1 | tee "$GPU_RESULTS_ROOT/compile.log"

CSV_FILE="$GPU_RESULTS_ROOT/all_gpu_team_sweep_results.csv"
echo "requested_teams,thread_limit,mapping,runs,mean_time_s,min_time_s,max_time_s,stddev_s,mean_actual_teams,max_actual_teams,mean_jacobi_iters,max_cpu_diff" > "$CSV_FILE"

for REQUESTED_TEAMS in "${TEAM_COUNTS[@]}"; do
  CONFIG_DIR="$GPU_RESULTS_ROOT/teams_${REQUESTED_TEAMS}"
  mkdir -p "$CONFIG_DIR/logs"
  RUN_TIMES=()
  RUN_ACTUAL_TEAMS=()
  RUN_ITERS=()
  RUN_DIFFS=()

  echo "=========================================================="
  echo "Configuration: requested teams=$REQUESTED_TEAMS thread_limit=$THREAD_LIMIT mapping=collapse2"
  for ((run = 1; run <= RUNS; run++)); do
    LOG_FILE="$CONFIG_DIR/logs/run${run}_$(date +'%Y%m%d_%H%M%S').log"
    echo "Run $run / $RUNS ..."
    OMP_NUM_TEAMS="$REQUESTED_TEAMS" OMP_TEAMS_THREAD_LIMIT="$THREAD_LIMIT" \
      OMP_TARGET_OFFLOAD=MANDATORY GPU_RESULTS_DIR="$CONFIG_DIR" \
      ./bin/heat_openmp_gpu_team_sweep "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" \
      > "$LOG_FILE" 2>&1

    if ! grep -q "Selected target.*GPU confirmed" "$LOG_FILE" ||
       ! grep -q "CPU baseline check.*PASSED" "$LOG_FILE"; then
      cat "$LOG_FILE"
      echo "Error: GPU execution or CPU validation failed in $LOG_FILE" >&2
      exit 1
    fi

    TIME_VAL="$(awk -F: '/^Execution time/ {gsub(/ seconds/, "", $2); gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
    TEAM_VAL="$(awk -F: '/^Max teams observed/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
    ITER_VAL="$(awk -F: '/^Total Jacobi iters/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
    DIFF_VAL="$(awk -F: '/^Max difference vs CPU baseline/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
    if [ -z "$TIME_VAL" ] || [ -z "$TEAM_VAL" ] || [ -z "$ITER_VAL" ] || [ -z "$DIFF_VAL" ]; then
      cat "$LOG_FILE"
      echo "Error: missing timing, actual team count, iteration count, or validation difference." >&2
      exit 1
    fi
    echo "Requested teams: $REQUESTED_TEAMS; observed teams: $TEAM_VAL"
    RUN_TIMES+=("$TIME_VAL")
    RUN_ACTUAL_TEAMS+=("$TEAM_VAL")
    RUN_ITERS+=("$ITER_VAL")
    RUN_DIFFS+=("$DIFF_VAL")
  done

  STATS="$(awk -v n="${#RUN_TIMES[@]}" '{ v[NR]=$1; sum+=$1 }
    END { mean=sum/n; min=v[1]; max=v[1]; ss=0;
      for (i=1; i<=n; i++) { if (v[i]<min) min=v[i]; if (v[i]>max) max=v[i]; d=v[i]-mean; ss+=d*d }
      sd=(n>1) ? sqrt(ss/(n-1)) : 0; printf "%.6f %.6f %.6f %.6f\n", mean,min,max,sd;
    }' < <(printf '%s\n' "${RUN_TIMES[@]}") )"
  read -r MEAN_TIME MIN_TIME MAX_TIME STDDEV_TIME <<< "$STATS"
  MEAN_TEAMS="$(awk '{sum+=$1} END {printf "%.3f", sum/NR}' < <(printf '%s\n' "${RUN_ACTUAL_TEAMS[@]}"))"
  MAX_TEAMS="$(awk 'NR==1 || $1>max {max=$1} END {print max}' < <(printf '%s\n' "${RUN_ACTUAL_TEAMS[@]}"))"
  MEAN_ITERS="$(awk '{sum+=$1} END {printf "%.3f", sum/NR}' < <(printf '%s\n' "${RUN_ITERS[@]}"))"
  MAX_DIFF="$(awk 'NR==1 || $1>max {max=$1} END {printf "%.9e", max}' < <(printf '%s\n' "${RUN_DIFFS[@]}"))"

  {
    echo "Requested teams: $REQUESTED_TEAMS"
    echo "Thread limit: $THREAD_LIMIT"
    echo "Mapping: collapse(2)"
    echo "Mean actual teams: $MEAN_TEAMS"
    echo "Maximum actual teams: $MAX_TEAMS"
    echo "Runs: $RUNS"
    echo "Mean time: $MEAN_TIME s"
    echo "Min time: $MIN_TIME s"
    echo "Max time: $MAX_TIME s"
    echo "Sample stddev: $STDDEV_TIME s"
    echo "Mean Jacobi iterations: $MEAN_ITERS"
    echo "Max CPU difference: $MAX_DIFF"
    echo "Logs: $CONFIG_DIR/logs/"
  } | tee "$CONFIG_DIR/summary.txt"

  echo "$REQUESTED_TEAMS,$THREAD_LIMIT,collapse2,$RUNS,$MEAN_TIME,$MIN_TIME,$MAX_TIME,$STDDEV_TIME,$MEAN_TEAMS,$MAX_TEAMS,$MEAN_ITERS,$MAX_DIFF" >> "$CSV_FILE"
done

echo "=========================================================="
echo "GPU team-count sweep complete."
echo "CSV: $CSV_FILE"
