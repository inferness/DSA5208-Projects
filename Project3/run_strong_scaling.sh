#!/usr/bin/env bash
# Strong-scaling sweep for the row-decomposed, nonblocking MPI implementation.
# Environment overrides: SCALING_N, SCALING_STEPS, SCALING_RUNS, SCALING_PROCS.

set -euo pipefail

N="${SCALING_N:-8192}"
STEPS="${SCALING_STEPS:-1000}"
RUNS="${SCALING_RUNS:-5}"
LAMBDA="${SCALING_LAMBDA:-0.05}"
MAX_ITERS="${SCALING_MAX_ITERS:-25}"
TOL="${SCALING_TOL:-1e-5}"
PROCS_LIST="${SCALING_PROCS:-1 4 16 32 64 128 256}"
MPI_CXX="${MPI_CXX:-mpicxx}"
MPI_RUN="${MPI_RUN:-mpirun}"
RESULTS_DIR="${RESULTS_DIR:-results/strong_scaling}"
BIN_DIR="${BIN_DIR:-bin/strong_scaling}"
SOURCE="heat_mpi_nonblocking_implicit.cpp"
BINARY="$BIN_DIR/heat_mpi_nonblocking_implicit"
CSV_FILE="$RESULTS_DIR/all_strong_scaling_results.csv"

if ! [[ "$N" =~ ^[1-9][0-9]*$ && "$STEPS" =~ ^[0-9]+$ && "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: N, steps, and repetitions must be valid positive/nonnegative integers." >&2
  exit 2
fi
for tool in "$MPI_CXX" "$MPI_RUN"; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Error: required command '$tool' not found." >&2
    exit 2
  fi
done

mkdir -p "$RESULTS_DIR/logs" "$BIN_DIR"
read -r -a PROCS <<< "$PROCS_LIST"
for p in "${PROCS[@]}"; do
  if ! [[ "$p" =~ ^[1-9][0-9]*$ ]] || (( p > N )); then
    echo "Error: invalid MPI rank count '$p' for N=$N." >&2
    exit 2
  fi
done

PRIOR_CSV="${SCALING_PRIOR_CSV:-}"
PRESERVED_ROWS="$RESULTS_DIR/.preserved_low_rank_rows.$$"
BASELINE_MEAN="${SCALING_BASELINE_TIME:-}"
HAS_P1=false
HAS_P4=false
for p in "${PROCS[@]}"; do
  [ "$p" = "1" ] && HAS_P1=true
  [ "$p" = "4" ] && HAS_P4=true
done
if [ "$HAS_P1" = false ]; then
  if [ -z "$PRIOR_CSV" ] || [ ! -r "$PRIOR_CSV" ]; then
    echo "Error: rerunning without P=1 requires SCALING_PRIOR_CSV with completed 1-rank results." >&2
    exit 2
  fi
  if [ -z "$BASELINE_MEAN" ]; then
    BASELINE_MEAN="$(awk -F, -v n="$N" 'NR>1 && $1==n && $3==1 && $5 ~ /^[0-9.]+$/ {sum+=$5; count++} END {if(count) printf "%.9f", sum/count}' "$PRIOR_CSV")"
  fi
  if [ -z "$BASELINE_MEAN" ]; then
    echo "Error: no completed P=1 timing for N=$N found in $PRIOR_CSV." >&2
    exit 2
  fi
  if [ "$HAS_P4" = false ]; then
    awk -F, -v n="$N" 'NR>1 && $1==n && $3==1 {print}' "$PRIOR_CSV" > "$PRESERVED_ROWS"
    awk -F, -v n="$N" 'NR>1 && $1==n && $3==4 {print}' "$PRIOR_CSV" >> "$PRESERVED_ROWS"
  fi
fi

# This Vanda Open MPI module does not appear to have its PBS/TM integration
# library available. Explicitly pass the PBS allocation so mpirun sees slots
# on every allocated node rather than only the local node.
MPI_HOST_ARGS=()
if [ -n "${PBS_NODEFILE:-}" ] && [ -r "$PBS_NODEFILE" ]; then
  HOSTFILE="$RESULTS_DIR/pbs_hosts.txt"
  awk '{ if (!($1 in slots)) order[++n]=$1; slots[$1]++ }
       END { for (i=1; i<=n; i++) printf "%s slots=%d\n", order[i], slots[order[i]] }' \
    "$PBS_NODEFILE" > "$HOSTFILE"
  MPI_HOST_ARGS=(--hostfile "$HOSTFILE")
fi

{
  echo "Strong-scaling sweep started: $(date)"
  echo "Host: $(hostname)"
  echo "PBS job ID: ${PBS_JOBID:-not set}"
  echo "N=$N steps=$STEPS lambda=$LAMBDA max_iters=$MAX_ITERS tol=$TOL"
  echo "MPI rank counts: ${PROCS[*]}"
  if [ -n "$BASELINE_MEAN" ]; then echo "P=1 baseline mean: $BASELINE_MEAN seconds"; fi
  echo "Repetitions per rank count: $RUNS"
  echo "MPI compiler: $MPI_CXX"
  "$MPI_CXX" --version 2>&1 | head -n 2 || true
  echo "MPI launcher: $MPI_RUN"
  "$MPI_RUN" --version 2>&1 | head -n 3 || true
  echo "Node and CPU details:"
  hostname -f 2>&1 || true
  lscpu 2>&1 || true
  echo "Allocated PBS slots: ${PBS_NODEFILE:-not set}"
  if [ -n "${PBS_NODEFILE:-}" ]; then sort "$PBS_NODEFILE" | uniq -c; fi
  if [ -n "${HOSTFILE:-}" ]; then
    echo "Open MPI hostfile (explicit PBS allocation): $HOSTFILE"
    cat "$HOSTFILE"
  fi
  echo "Loaded modules:"
  module list 2>&1 || true
} | tee "$RESULTS_DIR/toolchain.txt"

echo "Compiling $SOURCE ..."
"$MPI_CXX" -O3 -std=c++17 "$SOURCE" -o "$BINARY"

CSV_HEADER="N,steps,mpi_ranks,run,execution_time_s,speedup,efficiency,avg_rank_comp_time_s,max_rank_comp_time_s,avg_rank_comm_time_s,max_rank_comm_time_s,total_jacobi_iters,max_residual"
echo "$CSV_HEADER" > "$CSV_FILE"
if [ -f "$PRESERVED_ROWS" ]; then
  cat "$PRESERVED_ROWS" >> "$CSV_FILE"
  rm -f -- "$PRESERVED_ROWS"
fi

for P in "${PROCS[@]}"; do
  RUN_TIMES=()
  RUN_AVG_COMP=()
  RUN_MAX_COMP=()
  RUN_AVG_COMM=()
  RUN_MAX_COMM=()
  RUN_ITERS=()
  RUN_RESIDUALS=()

  echo "=========================================================="
  echo "Strong scaling: N=$N, MPI ranks=$P, runs=$RUNS"
  for ((run=1; run<=RUNS; run++)); do
    LOG_FILE="$RESULTS_DIR/logs/N${N}_P${P}_run${run}_$(date +'%Y%m%d_%H%M%S').log"
    echo "Run $run / $RUNS ..."
    "$MPI_RUN" "${MPI_HOST_ARGS[@]}" -np "$P" --map-by node --bind-to core --report-bindings \
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
    echo "  elapsed=$TIME_VAL s; avg/max rank compute=$AVG_COMP_VAL/$MAX_COMP_VAL s; avg/max rank comm=$AVG_COMM_VAL/$MAX_COMM_VAL s"
  done

  P_MEAN="$(printf '%s\n' "${RUN_TIMES[@]}" | awk '{sum+=$1} END {printf "%.9f", sum/NR}')"
  if [ "$P" -eq 1 ]; then BASELINE_MEAN="$P_MEAN"; fi
  for ((i=0; i<${#RUN_TIMES[@]}; i++)); do
    SPEEDUP="$(awk -v b="$BASELINE_MEAN" -v t="${RUN_TIMES[$i]}" 'BEGIN {if(t>0) printf "%.6f", b/t; else print "NA"}')"
    EFFICIENCY="$(awk -v s="$SPEEDUP" -v p="$P" 'BEGIN {if(s!="NA" && p>0) printf "%.6f", s/p; else print "NA"}')"
    echo "$N,$STEPS,$P,$((i+1)),${RUN_TIMES[$i]},$SPEEDUP,$EFFICIENCY,${RUN_AVG_COMP[$i]},${RUN_MAX_COMP[$i]},${RUN_AVG_COMM[$i]},${RUN_MAX_COMM[$i]},${RUN_ITERS[$i]},${RUN_RESIDUALS[$i]}" >> "$CSV_FILE"
  done

  {
    echo "N=$N steps=$STEPS MPI ranks=$P runs=$RUNS"
    echo "Mean execution time: $P_MEAN seconds"
    if [ -n "$BASELINE_MEAN" ]; then
      echo "Mean strong-scaling speedup: $(awk -v b="$BASELINE_MEAN" -v t="$P_MEAN" 'BEGIN {printf "%.6f",b/t}')"
      echo "Mean parallel efficiency: $(awk -v b="$BASELINE_MEAN" -v t="$P_MEAN" -v p="$P" 'BEGIN {printf "%.6f",b/(t*p)}')"
    fi
    echo "Per-run data: $CSV_FILE"
  } > "$RESULTS_DIR/N${N}_P${P}_summary.txt"
done

echo "=========================================================="
echo "Strong-scaling sweep finished: $(date)"
echo "CSV: $CSV_FILE"
echo "Run logs: $RESULTS_DIR/logs/"
