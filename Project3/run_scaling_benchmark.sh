#!/usr/bin/env bash
# Compile and record each repetition as an individual strong/weak-scaling row.
# Usage: ./run_scaling_benchmark.sh <source.cpp> <mpi_ranks> <runs> <N> <steps> [lambda=0.05] [max_iters=25] [tol=1e-5]

set -euo pipefail

SOURCE="${1:-}"
MPI_RANKS="${2:-1}"
RUNS="${3:-5}"
N="${4:-3000}"
STEPS="${5:-1000}"
LAMBDA="${6:-0.05}"
MAX_ITERS="${7:-25}"
TOL="${8:-1e-5}"
MPI_CXX="${MPI_CXX:-mpicxx}"
MPI_RUN="${MPI_RUN:-mpirun}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOGS_DIR="$RESULTS_DIR/logs"
BIN_DIR="${BIN_DIR:-bin}"

if [ ! -f "$SOURCE" ]; then
  echo "Error: source file '$SOURCE' not found." >&2
  exit 2
fi
if ! [[ "$MPI_RANKS" =~ ^[1-9][0-9]*$ && "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: MPI ranks and repetitions must be positive integers." >&2
  exit 2
fi
for tool in "$MPI_CXX" "$MPI_RUN"; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Error: required command '$tool' not found." >&2
    exit 2
  fi
done

mkdir -p "$RESULTS_DIR" "$LOGS_DIR" "$BIN_DIR"
BASE_NAME="$(basename "$SOURCE" .cpp)"
BINARY="$BIN_DIR/$BASE_NAME"
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
EXPERIMENT_ID="${EXPERIMENT_ID:-${BASE_NAME}_p${MPI_RANKS}}"
CSV_FILE="$RESULTS_DIR/all_results.csv"
SUMMARY_FILE="$RESULTS_DIR/${EXPERIMENT_ID}_summary.txt"
CSV_HEADER="timestamp,experiment,source_file,mpi_ranks,run,execution_time_s,speedup,efficiency,avg_rank_comp_time_s,max_rank_comp_time_s,avg_rank_comm_time_s,max_rank_comm_time_s"
if [ -f "$CSV_FILE" ]; then
  EXISTING_HEADER="$(head -n 1 "$CSV_FILE")"
  if [ "$EXISTING_HEADER" != "$CSV_HEADER" ]; then
    echo "Error: $CSV_FILE has an older/different schema. Start with a clean results/ directory." >&2
    exit 2
  fi
else
  echo "$CSV_HEADER" > "$CSV_FILE"
fi

echo "=========================================================="
echo "Experiment: $EXPERIMENT_ID"
echo "Source: $SOURCE | MPI ranks: $MPI_RANKS | repetitions: $RUNS"
echo "Problem: N=$N steps=$STEPS lambda=$LAMBDA max_iters=$MAX_ITERS tol=$TOL"
echo "Compiler: $MPI_CXX"
"$MPI_CXX" --version 2>&1 | head -n 2 || true
echo "=========================================================="

"$MPI_CXX" -O3 -std=c++17 "$SOURCE" -o "$BINARY"

RUN_TIMES=()
RUN_AVG_COMP=()
RUN_MAX_COMP=()
RUN_AVG_COMM=()
RUN_MAX_COMM=()
SERIAL_TIME_FILE="$RESULTS_DIR/optimized_serial_baseline_time.txt"
SOURCE_BASE="$(basename "$SOURCE")"

for ((run=1; run<=RUNS; run++)); do
  LOG_FILE="$LOGS_DIR/${EXPERIMENT_ID}_run${run}_${TIMESTAMP}.log"
  echo "Run $run / $RUNS ..."
  if grep -q '<mpi.h>' "$SOURCE"; then
    if [ "$MPI_RANKS" -eq 1 ]; then
      "$MPI_RUN" -np 1 --bind-to core "$BINARY" "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" > "$LOG_FILE" 2>&1
    else
      "$MPI_RUN" --bind-to none --map-by core -np "$MPI_RANKS" \
        "$BINARY" "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" > "$LOG_FILE" 2>&1
    fi
  else
    "$MPI_RUN" -np 1 --bind-to core "$BINARY" "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" > "$LOG_FILE" 2>&1
  fi

  TIME_VAL="$(grep -i '^Execution time' "$LOG_FILE" | head -n 1 | grep -oE '[0-9]+\.[0-9]+' | head -n 1 || true)"
  AVG_COMP_VAL="$(awk -F: 'tolower($1) ~ /average computation time/ {gsub(/ seconds.*/, "", $2); gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "$LOG_FILE")"
  MAX_COMP_VAL="$(awk -F: 'tolower($1) ~ /maximum computation time/ {gsub(/ seconds.*/, "", $2); gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "$LOG_FILE")"
  AVG_COMM_VAL="$(awk -F: 'tolower($1) ~ /average communication time/ {gsub(/ seconds.*/, "", $2); gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "$LOG_FILE")"
  MAX_COMM_VAL="$(awk -F: 'tolower($1) ~ /maximum communication time/ {gsub(/ seconds.*/, "", $2); gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "$LOG_FILE")"
  if [ -z "$TIME_VAL" ]; then
    cat "$LOG_FILE"
    echo "Error: no execution time found in $LOG_FILE" >&2
    exit 1
  fi
  AVG_COMP_VAL="${AVG_COMP_VAL:-NA}"
  MAX_COMP_VAL="${MAX_COMP_VAL:-NA}"
  AVG_COMM_VAL="${AVG_COMM_VAL:-NA}"
  MAX_COMM_VAL="${MAX_COMM_VAL:-NA}"

  SPEEDUP="N/A"
  EFFICIENCY="N/A"
  if [ "$SOURCE_BASE" = "heat_serial_implicit_optimized.cpp" ]; then
    SPEEDUP="1.00"
    EFFICIENCY="100.00%"
  elif [ "$SOURCE_BASE" != "heat_serial_implicit_unoptimized.cpp" ] && [ -f "$SERIAL_TIME_FILE" ]; then
    SERIAL_TIME="$(tr -d '[:space:]' < "$SERIAL_TIME_FILE")"
    SPEEDUP="$(awk -v s="$SERIAL_TIME" -v t="$TIME_VAL" 'BEGIN {if(t>0) printf "%.2f",s/t; else print "N/A"}')"
    EFFICIENCY="$(awk -v s="$SERIAL_TIME" -v t="$TIME_VAL" -v p="$MPI_RANKS" 'BEGIN {if(t>0&&p>0) printf "%.2f%%",100*s/(t*p); else print "N/A"}')"
  fi

  echo "${TIMESTAMP},${EXPERIMENT_ID},${SOURCE},${MPI_RANKS},${run},${TIME_VAL},${SPEEDUP},${EFFICIENCY},${AVG_COMP_VAL},${MAX_COMP_VAL},${AVG_COMM_VAL},${MAX_COMM_VAL}" >> "$CSV_FILE"
  RUN_TIMES+=("$TIME_VAL")
  RUN_AVG_COMP+=("$AVG_COMP_VAL")
  RUN_MAX_COMP+=("$MAX_COMP_VAL")
  RUN_AVG_COMM+=("$AVG_COMM_VAL")
  RUN_MAX_COMM+=("$MAX_COMM_VAL")
  echo "  elapsed=$TIME_VAL s | avg/max comp=$AVG_COMP_VAL/$MAX_COMP_VAL s | avg/max comm=$AVG_COMM_VAL/$MAX_COMM_VAL s"
done

mean_numeric() {
  printf '%s\n' "$@" | awk '$1 != "NA" {sum += $1; n++} END {if(n) printf "%.6f",sum/n; else print "NA"}'
}
median_numeric() {
  printf '%s\n' "$@" | awk '$1 != "NA"' | sort -n | awk '{v[NR]=$1} END {if(!NR) print "NA"; else if(NR%2) printf "%.6f",v[(NR+1)/2]; else printf "%.6f",(v[NR/2]+v[NR/2+1])/2}'
}
{
  echo "Experiment: $EXPERIMENT_ID"
  echo "Source: $SOURCE"
  echo "MPI ranks: $MPI_RANKS"
  echo "Problem: N=$N, steps=$STEPS, lambda=$LAMBDA, max_iters=$MAX_ITERS, tol=$TOL"
  echo "Per-run values (seconds):"
  for ((i=0; i<${#RUN_TIMES[@]}; i++)); do
    printf "  run=%d elapsed=%s avg_rank_comp=%s max_rank_comp=%s avg_rank_comm=%s max_rank_comm=%s\n" \
      "$((i+1))" "${RUN_TIMES[$i]}" "${RUN_AVG_COMP[$i]}" "${RUN_MAX_COMP[$i]}" "${RUN_AVG_COMM[$i]}" "${RUN_MAX_COMM[$i]}"
  done
  echo "Mean execution: $(mean_numeric "${RUN_TIMES[@]}") s"
  echo "Median execution: $(median_numeric "${RUN_TIMES[@]}") s"
  echo "Mean of per-run average-rank computation: $(mean_numeric "${RUN_AVG_COMP[@]}") s"
  echo "Mean of per-run maximum-rank computation: $(mean_numeric "${RUN_MAX_COMP[@]}") s"
  echo "Mean of per-run average-rank communication: $(mean_numeric "${RUN_AVG_COMM[@]}") s"
  echo "Mean of per-run maximum-rank communication: $(mean_numeric "${RUN_MAX_COMM[@]}") s"
  echo "Raw per-run CSV: $CSV_FILE"
  echo "Run logs: $LOGS_DIR/"
} > "$SUMMARY_FILE"

if [ "$SOURCE_BASE" = "heat_serial_implicit_optimized.cpp" ]; then
  mean_numeric "${RUN_TIMES[@]}" > "$SERIAL_TIME_FILE"
fi

echo "Summary: $SUMMARY_FILE"
