#!/usr/bin/env bash
# Compare persistent, per-time-step, and per-Jacobi-iteration GPU data mapping.
# Usage: ./run_gpu_mapping_sweep.sh [runs=5] [N=3000] [steps=1000] [lambda=0.05] [max_iters=25] [tol=1e-5]

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
RESULTS_ROOT="${GPU_RESULTS_DIR:-gpu_mapping_results}"
POLICIES=(persistent per_time_step per_jacobi_iteration)
SOURCES=(heat_openmp_gpu_offload.cpp heat_openmp_gpu_mapping_per_step.cpp heat_openmp_gpu_mapping_per_jacobi.cpp)
BINARIES=(heat_gpu_mapping_persistent heat_gpu_mapping_per_step heat_gpu_mapping_per_jacobi)
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

mkdir -p bin results/logs "$RESULTS_ROOT"
read -r -a GPU_FLAGS <<< "$GPU_CXXFLAGS"
export OMP_TARGET_OFFLOAD=MANDATORY
export OMP_TEAMS_THREAD_LIMIT="$THREAD_LIMIT"
unset OMP_NUM_TEAMS

{
  echo "OpenMP GPU data-mapping comparison started: $(date)"
  echo "Node: $(hostname)"
  echo "GPU compiler: $GPU_CXX"
  "$GPU_CXX" --version 2>&1 || true
  echo "CPU reference compiler: $CPU_CXX"
  "$CPU_CXX" --version 2>&1 | head -n 2 || true
  echo "GPU inventory:"
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L 2>&1 || true; else echo "nvidia-smi not found"; fi
  echo "Grid/steps: N=$N STEPS=$STEPS; runs/policy=$RUNS"
  echo "Kernel mapping: collapse(2)"
  echo "Team assignment: automatic"
  echo "OMP_TEAMS_THREAD_LIMIT: $THREAD_LIMIT"
  echo "Policies: ${POLICIES[*]}"
} | tee "$RESULTS_ROOT/gpu_mapping_toolchain.txt"

echo "[1/4] Compiling optimized CPU reference..."
"$CPU_CXX" -O3 -std=c++17 heat_serial_implicit_optimized.cpp -o bin/heat_serial_implicit_optimized
CPU_REFERENCE_LOG="$RESULTS_ROOT/heat_serial_implicit_optimized_reference.log"
OMP_NUM_THREADS=1 ./bin/heat_serial_implicit_optimized \
  "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" > "$CPU_REFERENCE_LOG" 2>&1

echo "[2/4] Compiling three GPU data-mapping variants..."
for i in "${!POLICIES[@]}"; do
  "$GPU_CXX" "${GPU_FLAGS[@]}" "${SOURCES[$i]}" -o "bin/${BINARIES[$i]}" \
    2>&1 | tee "$RESULTS_ROOT/compile_${POLICIES[$i]}.log"
done

CSV_FILE="$RESULTS_ROOT/all_gpu_mapping_results.csv"
echo "mapping_policy,requested_teams,thread_limit,runs,mean_time_s,min_time_s,max_time_s,stddev_s,mean_actual_teams,max_actual_teams,mean_jacobi_iters,max_cpu_diff" > "$CSV_FILE"

for i in "${!POLICIES[@]}"; do
  POLICY="${POLICIES[$i]}"
  BINARY="./bin/${BINARIES[$i]}"
  CONFIG_DIR="$RESULTS_ROOT/$POLICY"
  mkdir -p "$CONFIG_DIR/logs"
  RUN_TIMES=()
  RUN_TEAMS=()
  RUN_ITERS=()
  RUN_DIFFS=()

  echo "=========================================================="
  echo "Configuration: policy=$POLICY teams=automatic thread_limit=$THREAD_LIMIT collapse(2)"
  for ((run = 1; run <= RUNS; run++)); do
    LOG_FILE="$CONFIG_DIR/logs/run${run}_$(date +'%Y%m%d_%H%M%S').log"
    echo "Run $run / $RUNS ..."
    OMP_TEAMS_THREAD_LIMIT="$THREAD_LIMIT" OMP_TARGET_OFFLOAD=MANDATORY \
      GPU_RESULTS_DIR="$CONFIG_DIR" \
      "$BINARY" "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" > "$LOG_FILE" 2>&1

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
      echo "Error: missing timing, actual team count, iterations, or validation difference." >&2
      exit 1
    fi
    echo "Actual teams observed: $TEAM_VAL"
    RUN_TIMES+=("$TIME_VAL")
    RUN_TEAMS+=("$TEAM_VAL")
    RUN_ITERS+=("$ITER_VAL")
    RUN_DIFFS+=("$DIFF_VAL")
  done

  STATS="$(awk -v n="${#RUN_TIMES[@]}" '{ v[NR]=$1; sum+=$1 }
    END { mean=sum/n; min=v[1]; max=v[1]; ss=0;
      for (i=1; i<=n; i++) { if (v[i]<min) min=v[i]; if (v[i]>max) max=v[i]; d=v[i]-mean; ss+=d*d }
      sd=(n>1) ? sqrt(ss/(n-1)) : 0; printf "%.6f %.6f %.6f %.6f\n", mean,min,max,sd;
    }' < <(printf '%s\n' "${RUN_TIMES[@]}") )"
  read -r MEAN_TIME MIN_TIME MAX_TIME STDDEV_TIME <<< "$STATS"
  MEAN_TEAMS="$(awk '{sum+=$1} END {printf "%.3f", sum/NR}' < <(printf '%s\n' "${RUN_TEAMS[@]}"))"
  MAX_TEAMS="$(awk 'NR==1 || $1>max {max=$1} END {print max}' < <(printf '%s\n' "${RUN_TEAMS[@]}"))"
  MEAN_ITERS="$(awk '{sum+=$1} END {printf "%.3f", sum/NR}' < <(printf '%s\n' "${RUN_ITERS[@]}"))"
  MAX_DIFF="$(awk 'NR==1 || $1>max {max=$1} END {printf "%.9e", max}' < <(printf '%s\n' "${RUN_DIFFS[@]}"))"

  {
    echo "Mapping policy: $POLICY"
    echo "Requested team count: automatic"
    echo "Thread limit: $THREAD_LIMIT"
    echo "Loop mapping: collapse(2)"
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

  echo "$POLICY,automatic,$THREAD_LIMIT,$RUNS,$MEAN_TIME,$MIN_TIME,$MAX_TIME,$STDDEV_TIME,$MEAN_TEAMS,$MAX_TEAMS,$MEAN_ITERS,$MAX_DIFF" >> "$CSV_FILE"
done

echo "=========================================================="
echo "GPU data-mapping comparison complete."
echo "CSV: $CSV_FILE"
