#!/usr/bin/env bash
# Sweep N and OpenMP team settings on one GPU using collapse(2).
# Usage: ./run_gpu_team_sweep.sh [runs=5] [steps=1000] [lambda=0.05] [max_iters=25] [tol=1e-5]
# Override grid sizes with GPU_N_SIZES="1024 4096 8192 16000".

set -euo pipefail

RUNS="${1:-5}"
STEPS="${2:-1000}"
LAMBDA="${3:-0.05}"
MAX_ITERS="${4:-25}"
TOL="${5:-1e-5}"
N_SIZES="${GPU_N_SIZES:-1024 4096 8192 16000}"
GPU_CXX="${GPU_CXX:-nvc++}"
CPU_CXX="${CPU_CXX:-g++}"
GPU_CXXFLAGS="${GPU_CXXFLAGS:--O3 -std=c++17 -mp=gpu -Minfo=mp}"
GPU_RESULTS_ROOT="${GPU_RESULTS_DIR:-results/openmp_gpu_size_team_sweep}"
BIN_DIR="${BIN_DIR:-bin/openmp_gpu_size_team_sweep}"
TEAM_SETTINGS=(auto 256 512 1024)
THREAD_LIMIT=256

if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ && "$STEPS" =~ ^[0-9]+$ ]]; then
  echo "Error: runs must be positive and steps must be nonnegative integers." >&2
  exit 2
fi
for compiler in "$GPU_CXX" "$CPU_CXX"; do
  if ! command -v "$compiler" >/dev/null 2>&1; then
    echo "Error: compiler '$compiler' not found." >&2
    exit 2
  fi
done

read -r -a N_ARRAY <<< "$N_SIZES"
if [ "${#N_ARRAY[@]}" -eq 0 ]; then
  echo "Error: GPU_N_SIZES is empty." >&2
  exit 2
fi
for N in "${N_ARRAY[@]}"; do
  if ! [[ "$N" =~ ^[1-9][0-9]*$ ]] || (( N < 3 )); then
    echo "Error: invalid grid size '$N'; N must be an integer >= 3." >&2
    exit 2
  fi
done

mkdir -p "$BIN_DIR" "$GPU_RESULTS_ROOT"
read -r -a GPU_FLAGS <<< "$GPU_CXXFLAGS"
export OMP_TARGET_OFFLOAD=MANDATORY
export OMP_TEAMS_THREAD_LIMIT="$THREAD_LIMIT"
unset OMP_NUM_TEAMS

{
  echo "OpenMP GPU size/team sweep started: $(date)"
  echo "Node: $(hostname)"
  echo "GPU compiler: $GPU_CXX"
  "$GPU_CXX" --version 2>&1 || true
  echo "CPU reference compiler: $CPU_CXX"
  "$CPU_CXX" --version 2>&1 | head -n 2 || true
  echo "GPU inventory:"
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L 2>&1 || true; else echo "nvidia-smi not found"; fi
  echo "Loaded modules:"
  module list 2>&1 || true
  echo "Grid sizes: ${N_ARRAY[*]}"
  echo "Steps=$STEPS lambda=$LAMBDA max_iters=$MAX_ITERS tol=$TOL; runs/configuration=$RUNS"
  echo "Loop mapping: collapse(2)"
  echo "OMP_TEAMS_THREAD_LIMIT: $THREAD_LIMIT"
  echo "Team settings: automatic, 256, 512, 1024"
  echo "Automatic mode leaves OMP_NUM_TEAMS unset."
} | tee "$GPU_RESULTS_ROOT/gpu_size_team_sweep_toolchain.txt"

echo "[1/3] Compiling optimized CPU reference..."
"$CPU_CXX" -O3 -std=c++17 heat_serial_implicit_optimized.cpp \
  -o "$BIN_DIR/heat_serial_implicit_optimized"

echo "[2/3] Compiling collapse(2) GPU code with per-launch team reporting..."
"$GPU_CXX" "${GPU_FLAGS[@]}" heat_openmp_gpu_offload.cpp \
  -o "$BIN_DIR/heat_openmp_gpu_size_team_sweep" 2>&1 | tee "$GPU_RESULTS_ROOT/compile.log"

RAW_CSV="$GPU_RESULTS_ROOT/all_gpu_size_team_runs.csv"
AGG_CSV="$GPU_RESULTS_ROOT/all_gpu_size_team_aggregated.csv"
echo "N,steps,team_setting,requested_teams,thread_limit,mapping,run,execution_time_s,min_teams_observed,max_teams_observed,mean_teams_per_launch,team_launches_sampled,observed_team_count_values,total_jacobi_iters,max_cpu_diff,max_residual" > "$RAW_CSV"
echo "N,steps,team_setting,requested_teams,thread_limit,mapping,runs,mean_time_s,stddev_time_s,min_time_s,max_time_s,mean_teams_per_launch,min_teams_observed,max_teams_observed,teams_varied_within_run,team_count_varied_between_runs,mean_jacobi_iters,max_cpu_diff,max_residual" > "$AGG_CSV"

for N in "${N_ARRAY[@]}"; do
  N_DIR="$GPU_RESULTS_ROOT/N${N}"
  mkdir -p "$N_DIR"
  CPU_REFERENCE_LOG="$N_DIR/heat_serial_implicit_optimized_reference.log"
  BASELINE_BIN="$GPU_RESULTS_ROOT/baseline_implicit_N${N}_S${STEPS}.bin"

  echo "=========================================================="
  echo "Grid size N=$N: generating CPU validation baseline once."
  RESULTS_DIR="$GPU_RESULTS_ROOT" OMP_NUM_THREADS=1 \
    "$BIN_DIR/heat_serial_implicit_optimized" \
    "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" \
    > "$CPU_REFERENCE_LOG" 2>&1
  if [ ! -s "$BASELINE_BIN" ]; then
    cat "$CPU_REFERENCE_LOG"
    echo "Error: CPU baseline grid was not generated at $BASELINE_BIN" >&2
    exit 1
  fi

  for TEAM_SETTING in "${TEAM_SETTINGS[@]}"; do
    CONFIG_DIR="$N_DIR/teams_${TEAM_SETTING}"
    mkdir -p "$CONFIG_DIR/logs"
    RUN_TIMES=()
    RUN_TEAM_MINS=()
    RUN_TEAM_MAXES=()
    RUN_TEAM_MEANS=()
    RUN_LAUNCHES=()
    RUN_TEAM_SIGNATURES=()
    RUN_ITERS=()
    RUN_DIFFS=()
    RUN_RESIDUALS=()
    REQUESTED_TEAMS="$TEAM_SETTING"
    if [ "$TEAM_SETTING" = "auto" ]; then REQUESTED_TEAMS="automatic"; fi

    echo "----------------------------------------------------------"
    echo "N=$N team setting=$TEAM_SETTING thread limit=$THREAD_LIMIT mapping=collapse2"
    for ((run = 1; run <= RUNS; run++)); do
      LOG_FILE="$CONFIG_DIR/logs/run${run}_$(date +'%Y%m%d_%H%M%S').log"
      echo "Run $run / $RUNS ..."
      if [ "$TEAM_SETTING" = "auto" ]; then
        env -u OMP_NUM_TEAMS OMP_TEAMS_THREAD_LIMIT="$THREAD_LIMIT" \
          OMP_TARGET_OFFLOAD=MANDATORY RESULTS_DIR="$GPU_RESULTS_ROOT" \
          GPU_RESULTS_DIR="$CONFIG_DIR" GPU_SAVE_RESULT_GRID=0 \
          "$BIN_DIR/heat_openmp_gpu_size_team_sweep" \
          "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" > "$LOG_FILE" 2>&1
      else
        env OMP_NUM_TEAMS="$TEAM_SETTING" OMP_TEAMS_THREAD_LIMIT="$THREAD_LIMIT" \
          OMP_TARGET_OFFLOAD=MANDATORY RESULTS_DIR="$GPU_RESULTS_ROOT" \
          GPU_RESULTS_DIR="$CONFIG_DIR" GPU_SAVE_RESULT_GRID=0 \
          "$BIN_DIR/heat_openmp_gpu_size_team_sweep" \
          "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" > "$LOG_FILE" 2>&1
      fi

      if ! grep -q "Selected target.*GPU confirmed" "$LOG_FILE" ||
         ! grep -q "CPU baseline check.*PASSED" "$LOG_FILE"; then
        cat "$LOG_FILE"
        echo "Error: GPU execution or CPU validation failed in $LOG_FILE" >&2
        exit 1
      fi

      TIME_VAL="$(awk -F: '/^Execution time/ {gsub(/ seconds/, "", $2); gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
      TEAM_MIN="$(awk -F: '/^Min teams observed/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
      TEAM_MAX="$(awk -F: '/^Max teams observed/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
      TEAM_MEAN="$(awk -F: '/^Mean teams per launch/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
      LAUNCHES="$(awk -F: '/^Team launches sampled/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
      TEAM_SIGNATURE="$(awk -F: '/^Observed team count values/ {sub(/^[[:space:]]*/, "", $2); gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
      ITER_VAL="$(awk -F: '/^Total Jacobi iters/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
      DIFF_VAL="$(awk -F: '/^Max difference vs CPU baseline/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
      RESIDUAL_VAL="$(awk -F: '/^Max absolute residual/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
      if [ -z "$TIME_VAL" ] || [ -z "$TEAM_MIN" ] || [ -z "$TEAM_MAX" ] ||
         [ -z "$TEAM_MEAN" ] || [ -z "$LAUNCHES" ] || [ -z "$TEAM_SIGNATURE" ] || [ -z "$ITER_VAL" ] ||
         [ -z "$DIFF_VAL" ] || [ -z "$RESIDUAL_VAL" ]; then
        cat "$LOG_FILE"
        echo "Error: missing timing, team-count, iteration, or validation data in $LOG_FILE" >&2
        exit 1
      fi

      echo "$N,$STEPS,$TEAM_SETTING,$REQUESTED_TEAMS,$THREAD_LIMIT,collapse2,$run,$TIME_VAL,$TEAM_MIN,$TEAM_MAX,$TEAM_MEAN,$LAUNCHES,$TEAM_SIGNATURE,$ITER_VAL,$DIFF_VAL,$RESIDUAL_VAL" >> "$RAW_CSV"
      RUN_TIMES+=("$TIME_VAL")
      RUN_TEAM_MINS+=("$TEAM_MIN")
      RUN_TEAM_MAXES+=("$TEAM_MAX")
      RUN_TEAM_MEANS+=("$TEAM_MEAN")
      RUN_LAUNCHES+=("$LAUNCHES")
      RUN_TEAM_SIGNATURES+=("$TEAM_SIGNATURE")
      RUN_ITERS+=("$ITER_VAL")
      RUN_DIFFS+=("$DIFF_VAL")
      RUN_RESIDUALS+=("$RESIDUAL_VAL")
      echo "  time=$TIME_VAL s; teams min/max/mean per launch=$TEAM_MIN/$TEAM_MAX/$TEAM_MEAN; sampled launches=$LAUNCHES"
    done

    TIME_STATS="$(printf '%s\n' "${RUN_TIMES[@]}" | awk '
      { v[NR]=$1; sum+=$1 }
      END { mean=sum/NR; min=v[1]; max=v[1]; ss=0;
        for (i=1; i<=NR; i++) { if (v[i]<min) min=v[i]; if (v[i]>max) max=v[i]; d=v[i]-mean; ss+=d*d }
        sd=(NR>1) ? sqrt(ss/(NR-1)) : 0;
        printf "%.6f %.6f %.6f %.6f", mean,sd,min,max;
      }')"
    read -r MEAN_TIME STDDEV_TIME MIN_TIME MAX_TIME <<< "$TIME_STATS"
    TEAM_STATS="$(paste <(printf '%s\n' "${RUN_TEAM_MINS[@]}") \
      <(printf '%s\n' "${RUN_TEAM_MAXES[@]}") \
      <(printf '%s\n' "${RUN_TEAM_MEANS[@]}") \
      <(printf '%s\n' "${RUN_LAUNCHES[@]}") \
      <(printf '%s\n' "${RUN_TEAM_SIGNATURES[@]}") | awk '
        NR==1 { minall=$1; maxall=$2; firstsignature=$5 }
        { if ($1<minall) minall=$1; if ($2>maxall) maxall=$2;
          if ($1!=$2) within=1; if ($5!=firstsignature) between=1;
          weighted += $3*$4; launches += $4; }
        END { mean=(launches>0) ? weighted/launches : 0;
          printf "%.6f %d %d %d %d", mean,minall,maxall,within+0,between+0;
        }')"
    read -r MEAN_TEAMS MIN_TEAMS MAX_TEAMS WITHIN_VAR BETWEEN_VAR <<< "$TEAM_STATS"
    MEAN_ITERS="$(printf '%s\n' "${RUN_ITERS[@]}" | awk '{sum+=$1} END {printf "%.3f",sum/NR}')"
    MAX_DIFF="$(printf '%s\n' "${RUN_DIFFS[@]}" | awk 'NR==1 || $1>max {max=$1} END {printf "%.9e",max}')"
    MAX_RESIDUAL="$(printf '%s\n' "${RUN_RESIDUALS[@]}" | awk 'NR==1 || $1>max {max=$1} END {printf "%.9e",max}')"

    {
      echo "OpenMP GPU size/team configuration"
      echo "N=$N steps=$STEPS"
      echo "Team setting: $TEAM_SETTING (requested=$REQUESTED_TEAMS)"
      echo "Thread limit: $THREAD_LIMIT"
      echo "Mapping: collapse(2)"
      echo "Runs: $RUNS"
      echo "Mean time: $MEAN_TIME s"
      echo "Sample stddev: $STDDEV_TIME s"
      echo "Min/max time: $MIN_TIME / $MAX_TIME s"
      echo "Mean teams per launch: $MEAN_TEAMS"
      echo "Minimum/maximum teams observed: $MIN_TEAMS / $MAX_TEAMS"
      echo "Team count varied within any run: $([[ "$WITHIN_VAR" == 1 ]] && echo yes || echo no)"
      echo "Observed team-count signatures by run: ${RUN_TEAM_SIGNATURES[*]}"
      echo "Team-count choice varied between runs: $([[ "$BETWEEN_VAR" == 1 ]] && echo yes || echo no)"
      echo "Mean Jacobi iterations: $MEAN_ITERS"
      echo "Maximum CPU difference: $MAX_DIFF"
      echo "Maximum residual: $MAX_RESIDUAL"
      echo "Logs: $CONFIG_DIR/logs/"
    } | tee "$CONFIG_DIR/summary.txt"

    echo "$N,$STEPS,$TEAM_SETTING,$REQUESTED_TEAMS,$THREAD_LIMIT,collapse2,$RUNS,$MEAN_TIME,$STDDEV_TIME,$MIN_TIME,$MAX_TIME,$MEAN_TEAMS,$MIN_TEAMS,$MAX_TEAMS,$WITHIN_VAR,$BETWEEN_VAR,$MEAN_ITERS,$MAX_DIFF,$MAX_RESIDUAL" >> "$AGG_CSV"
  done

  # The GPU runs have all been checked against this baseline; remove the large
  # generated grid to keep the result folder compact, retaining its text log.
  rm -f -- "$BASELINE_BIN"
done

echo "=========================================================="
echo "GPU size/team sweep complete."
echo "Per-run CSV: $RAW_CSV"
echo "Aggregated CSV: $AGG_CSV"
echo "Detailed logs: $GPU_RESULTS_ROOT/N<N>/teams_<setting>/logs/"
