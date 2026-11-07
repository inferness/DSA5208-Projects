#!/usr/bin/env bash
# Weak-scaling sweep for row decomposition + nonblocking MPI.
# Each rank owns exactly the same number of grid cells for each configuration.
# Environment overrides: WEAK_CONFIGS, WEAK_STEPS, WEAK_RUNS, WEAK_LAMBDA,
# WEAK_MAX_ITERS, WEAK_TOL, RESULTS_DIR, BIN_DIR.

set -euo pipefail

CONFIGS="${WEAK_CONFIGS:-1:1024 4:2048 16:4096 32:5792 64:8192 128:11520 256:16384}"
STEPS="${WEAK_STEPS:-1000}"
RUNS="${WEAK_RUNS:-5}"
LAMBDA="${WEAK_LAMBDA:-0.05}"
MAX_ITERS="${WEAK_MAX_ITERS:-25}"
TOL="${WEAK_TOL:-1e-5}"
MPI_CXX="${MPI_CXX:-mpicxx}"
MPI_RUN="${MPI_RUN:-mpirun}"
RESULTS_DIR="${RESULTS_DIR:-results/weak_scaling}"
BIN_DIR="${BIN_DIR:-bin/weak_scaling}"
SOURCE="heat_mpi_nonblocking_implicit.cpp"
BINARY="$BIN_DIR/heat_mpi_nonblocking_implicit"
CSV_FILE="$RESULTS_DIR/all_weak_scaling_results.csv"

if ! [[ "$STEPS" =~ ^[0-9]+$ && "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: steps must be nonnegative and repetitions must be positive." >&2
  exit 2
fi
for tool in "$MPI_CXX" "$MPI_RUN"; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Error: required command '$tool' not found." >&2
    exit 2
  fi
done

mkdir -p "$RESULTS_DIR/logs" "$BIN_DIR"
read -r -a CONFIG_ARRAY <<< "$CONFIGS"
if [ "${#CONFIG_ARRAY[@]}" -eq 0 ]; then
  echo "Error: WEAK_CONFIGS is empty. Expected rank:N pairs." >&2
  exit 2
fi

for config in "${CONFIG_ARRAY[@]}"; do
  if ! [[ "$config" =~ ^[1-9][0-9]*:[1-9][0-9]*$ ]]; then
    echo "Error: invalid weak-scaling configuration '$config'; expected ranks:N." >&2
    exit 2
  fi
  p="${config%%:*}"
  n="${config##*:}"
  if (( n < 3 || n % p != 0 )); then
    echo "Error: N=$n must be >=3 and divisible by P=$p to assign equal rows." >&2
    exit 2
  fi
done

# The Open MPI module on Vanda may lack PBS/TM integration. Construct an
# explicit per-configuration hostfile using the minimum nodes needed. Rank
# order is filled by node so adjacent row ranks stay together on the same host.
PBS_HOSTS_FILE=""
if [ -n "${PBS_NODEFILE:-}" ] && [ -r "$PBS_NODEFILE" ]; then
  PBS_HOSTS_FILE="$RESULTS_DIR/pbs_allocation_hosts.txt"
  awk '{ if (!($1 in slots)) order[++n]=$1; slots[$1]++ }
       END { for (i=1; i<=n; i++) printf "%s slots=%d\n", order[i], slots[order[i]] }' \
    "$PBS_NODEFILE" > "$PBS_HOSTS_FILE"
fi

{
  echo "Weak-scaling sweep started: $(date)"
  echo "Host: $(hostname)"
  echo "PBS job ID: ${PBS_JOBID:-not set}"
  echo "Configurations (P:N): ${CONFIG_ARRAY[*]}"
  echo "Target owned cells/rank: 1048576"
  echo "Steps=$STEPS lambda=$LAMBDA max_iters=$MAX_ITERS tol=$TOL repetitions=$RUNS"
  echo "MPI compiler: $MPI_CXX"
  "$MPI_CXX" --version 2>&1 | head -n 2 || true
  echo "MPI launcher: $MPI_RUN"
  "$MPI_RUN" --version 2>&1 | head -n 3 || true
  echo "Allocated PBS slots: ${PBS_NODEFILE:-not set}"
  if [ -n "${PBS_NODEFILE:-}" ]; then sort "$PBS_NODEFILE" | uniq -c; fi
  if [ -n "$PBS_HOSTS_FILE" ]; then cat "$PBS_HOSTS_FILE"; fi
  echo "Node and CPU details:"
  lscpu 2>&1 || true
  echo "Loaded modules:"
  module list 2>&1 || true
} | tee "$RESULTS_DIR/toolchain.txt"

echo "Compiling $SOURCE ..."
"$MPI_CXX" -O3 -std=c++17 "$SOURCE" -o "$BINARY"

CSV_HEADER="N,steps,mpi_ranks,local_rows_per_rank,owned_cells_per_rank,min_active_cells_per_rank,max_active_cells_per_rank,target_cells_per_rank,target_error_pct,run,execution_time_s,weak_scaling_efficiency,avg_rank_comp_time_s,max_rank_comp_time_s,avg_rank_comm_time_s,max_rank_comm_time_s,total_jacobi_iters,mean_jacobi_iters_per_step,global_cell_updates_per_s,max_residual"
echo "$CSV_HEADER" > "$CSV_FILE"
BASELINE_MEAN=""

for config in "${CONFIG_ARRAY[@]}"; do
  P="${config%%:*}"
  N="${config##*:}"
  LOCAL_ROWS=$((N / P))
  OWNED_CELLS=$((LOCAL_ROWS * N))
  if [ "$P" -eq 1 ]; then
    MIN_ACTIVE_CELLS=$(((N - 2) * (N - 2)))
    MAX_ACTIVE_CELLS="$MIN_ACTIVE_CELLS"
  else
    MIN_ACTIVE_CELLS=$(((LOCAL_ROWS - 1) * (N - 2)))
    MAX_ACTIVE_CELLS=$((LOCAL_ROWS * (N - 2)))
  fi
  TARGET_CELLS=1048576
  TARGET_ERROR="$(awk -v a="$OWNED_CELLS" -v t="$TARGET_CELLS" 'BEGIN {printf "%.6f",100*(a-t)/t}')"
  MPI_HOST_ARGS=()
  if [ -n "$PBS_HOSTS_FILE" ]; then
    HOSTFILE="$RESULTS_DIR/pbs_hosts_P${P}.txt"
    if ! awk -v required="$P" 'BEGIN { remaining=required }
      {
        host=$1; available=$2; sub(/^slots=/, "", available)
        available += 0
        remaining += 0
        if (remaining > 0) {
          use=(available < remaining) ? available : remaining
          printf "%s slots=%d\n", host, use
          remaining -= use
        }
      }
      END { if (remaining > 0) exit 1 }
    ' "$PBS_HOSTS_FILE" > "$HOSTFILE"; then
      echo "Error: PBS allocation has fewer than $P slots." >&2
      exit 2
    fi
    MPI_HOST_ARGS=(--hostfile "$HOSTFILE")
    echo "MPI hostfile for P=$P:"
    cat "$HOSTFILE"
  fi

  RUN_TIMES=()
  RUN_AVG_COMP=()
  RUN_MAX_COMP=()
  RUN_AVG_COMM=()
  RUN_MAX_COMM=()
  RUN_ITERS=()
  RUN_RESIDUALS=()

  echo "=========================================================="
  echo "Weak scaling: P=$P N=$N rows/rank=$LOCAL_ROWS owned cells/rank=$OWNED_CELLS runs=$RUNS"
  for ((run=1; run<=RUNS; run++)); do
    LOG_FILE="$RESULTS_DIR/logs/N${N}_P${P}_run${run}_$(date +'%Y%m%d_%H%M%S').log"
    echo "Run $run / $RUNS ..."
    "$MPI_RUN" "${MPI_HOST_ARGS[@]}" -np "$P" --map-by slot --bind-to core --report-bindings \
      "$BINARY" "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" \
      > "$LOG_FILE" 2>&1

    TIME_VAL="$(awk -F: '/^Execution time/ {gsub(/ seconds.*/, "", $2); gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "$LOG_FILE")"
    AVG_COMP_VAL="$(awk -F: 'tolower($1) ~ /average computation time/ {gsub(/ seconds.*/, "", $2); gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "$LOG_FILE")"
    MAX_COMP_VAL="$(awk -F: 'tolower($1) ~ /maximum computation time/ {gsub(/ seconds.*/, "", $2); gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "$LOG_FILE")"
    AVG_COMM_VAL="$(awk -F: 'tolower($1) ~ /average communication time/ {gsub(/ seconds.*/, "", $2); gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "$LOG_FILE")"
    MAX_COMM_VAL="$(awk -F: 'tolower($1) ~ /maximum communication time/ {gsub(/ seconds.*/, "", $2); gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "$LOG_FILE")"
    ITERS_VAL="$(awk -F: '/^Total Jacobi iters/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
    RESIDUAL_VAL="$(awk -F: '/^Max absolute residual/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
    if [ -z "$TIME_VAL" ] || [ -z "$AVG_COMP_VAL" ] || [ -z "$MAX_COMP_VAL" ] ||
       [ -z "$AVG_COMM_VAL" ] || [ -z "$MAX_COMM_VAL" ] ||
       [ -z "$ITERS_VAL" ] || [ -z "$RESIDUAL_VAL" ]; then
      cat "$LOG_FILE"
      echo "Error: timing or validation fields missing in $LOG_FILE" >&2
      exit 1
    fi

    RUN_TIMES+=("$TIME_VAL")
    RUN_AVG_COMP+=("$AVG_COMP_VAL")
    RUN_MAX_COMP+=("$MAX_COMP_VAL")
    RUN_AVG_COMM+=("$AVG_COMM_VAL")
    RUN_MAX_COMM+=("$MAX_COMM_VAL")
    RUN_ITERS+=("$ITERS_VAL")
    RUN_RESIDUALS+=("$RESIDUAL_VAL")
    echo "  elapsed=$TIME_VAL s; compute avg/max=$AVG_COMP_VAL/$MAX_COMP_VAL s; comm avg/max=$AVG_COMM_VAL/$MAX_COMM_VAL s"
  done

  P_MEAN="$(printf '%s\n' "${RUN_TIMES[@]}" | awk '{sum+=$1} END {printf "%.9f",sum/NR}')"
  P_ITERS_MEAN="$(printf '%s\n' "${RUN_ITERS[@]}" | awk '{sum+=$1} END {printf "%.9f",sum/NR}')"
  if [ "$P" -eq 1 ]; then
    BASELINE_MEAN="$P_MEAN"
  fi
  for ((i=0; i<${#RUN_TIMES[@]}; i++)); do
    WEAK_EFFICIENCY="$(awk -v b="$BASELINE_MEAN" -v t="${RUN_TIMES[$i]}" 'BEGIN {if(t>0) printf "%.6f",b/t; else print "NA"}')"
    MEAN_ITERS_PER_STEP="$(awk -v i="${RUN_ITERS[$i]}" -v s="$STEPS" 'BEGIN {if(s>0) printf "%.6f",i/s; else print "NA"}')"
    CELL_UPDATES_PER_SEC="$(awk -v n="$N" -v it="${RUN_ITERS[$i]}" -v t="${RUN_TIMES[$i]}" 'BEGIN {if(t>0) printf "%.6e",((n-2)*(n-2)*it)/t; else print "NA"}')"
    echo "$N,$STEPS,$P,$LOCAL_ROWS,$OWNED_CELLS,$MIN_ACTIVE_CELLS,$MAX_ACTIVE_CELLS,$TARGET_CELLS,$TARGET_ERROR,$((i+1)),${RUN_TIMES[$i]},$WEAK_EFFICIENCY,${RUN_AVG_COMP[$i]},${RUN_MAX_COMP[$i]},${RUN_AVG_COMM[$i]},${RUN_MAX_COMM[$i]},${RUN_ITERS[$i]},$MEAN_ITERS_PER_STEP,$CELL_UPDATES_PER_SEC,${RUN_RESIDUALS[$i]}" >> "$CSV_FILE"
  done

  {
    echo "Weak-scaling configuration: P=$P N=$N steps=$STEPS"
    echo "Owned cells per rank: $OWNED_CELLS (target=$TARGET_CELLS; error=$TARGET_ERROR%)"
    echo "Active cells/rank range (accounts for fixed physical boundary rows): $MIN_ACTIVE_CELLS-$MAX_ACTIVE_CELLS"
    echo "Repetitions: $RUNS"
    echo "Mean execution time: $P_MEAN seconds"
    if [ -n "$BASELINE_MEAN" ]; then
      echo "Weak-scaling efficiency (T1/Tp): $(awk -v b="$BASELINE_MEAN" -v t="$P_MEAN" 'BEGIN {printf "%.6f",b/t}')"
    fi
    echo "Mean total Jacobi iterations: $P_ITERS_MEAN"
    echo "CSV: $CSV_FILE"
  } > "$RESULTS_DIR/N${N}_P${P}_summary.txt"
done

echo "=========================================================="
echo "Weak-scaling sweep finished: $(date)"
echo "CSV: $CSV_FILE"
echo "Run logs: $RESULTS_DIR/logs/"
