#!/usr/bin/env bash
# One-run large-grid pilot using the existing pure-MPI nonblocking code.

set -euo pipefail

N=8192
STEPS=1000
LAMBDA=0.05
MAX_ITERS=25
TOL=1e-5
MPI_CXX="${MPI_CXX:-mpicxx}"
MPI_RUN="${MPI_RUN:-mpirun}"
RESULTS_DIR="${SCALING_PILOT_DIR:-results/scaling_pilot}"
BIN_DIR="${BIN_DIR:-bin}"
P=1

for tool in "$MPI_CXX" "$MPI_RUN"; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Error: required command '$tool' not found." >&2
    exit 2
  fi
done

mkdir -p "$RESULTS_DIR"

{
  echo "Strong-scaling pilot: $(date)"
  echo "Host: $(hostname)"
  echo "Grid: N=$N; steps=$STEPS; lambda=$LAMBDA; max Jacobi iterations=$MAX_ITERS; tolerance=$TOL"
  echo "MPI ranks: $P"
  echo "Repetitions: 1"
  echo "MPI compiler: $MPI_CXX"
  "$MPI_CXX" --version 2>&1 | head -n 2 || true
  echo "MPI launcher: $MPI_RUN"
  "$MPI_RUN" --version 2>&1 | head -n 3 || true
  echo "CPU details:"
  lscpu 2>&1 || true
  echo "Loaded modules:"
  module list 2>&1 || true
} | tee "$RESULTS_DIR/toolchain.txt"

echo "Compiling heat_mpi_nonblocking_implicit.cpp..."
"$MPI_CXX" -O3 -std=c++17 heat_mpi_nonblocking_implicit.cpp \
  -o "$BIN_DIR/heat_mpi_nonblocking_implicit"

RUN_LOG="$RESULTS_DIR/heat_mpi_nonblocking_N${N}_P${P}_run1.log"
echo "Running the single pilot..."
"$MPI_RUN" -np "$P" --map-by ppr:1:node:PE=1 --bind-to core --report-bindings \
  "$BIN_DIR/heat_mpi_nonblocking_implicit" \
  "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" \
  > "$RUN_LOG" 2>&1

TIME_VAL="$(awk -F: '/^Execution time/ {gsub(/ seconds.*/, "", $2); gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "$RUN_LOG")"
AVG_COMP_VAL="$(awk -F: 'tolower($1) ~ /average computation time/ {gsub(/ seconds.*/, "", $2); gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "$RUN_LOG")"
MAX_COMP_VAL="$(awk -F: 'tolower($1) ~ /maximum computation time/ {gsub(/ seconds.*/, "", $2); gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "$RUN_LOG")"
AVG_COMM_VAL="$(awk -F: 'tolower($1) ~ /average communication time/ {gsub(/ seconds.*/, "", $2); gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "$RUN_LOG")"
MAX_COMM_VAL="$(awk -F: 'tolower($1) ~ /maximum communication time/ {gsub(/ seconds.*/, "", $2); gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "$RUN_LOG")"
ITERS_VAL="$(awk -F: '/^Total Jacobi iters/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$RUN_LOG")"
RESIDUAL_VAL="$(awk -F: '/^Max absolute residual/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$RUN_LOG")"

if [ -z "$TIME_VAL" ] || [ -z "$AVG_COMP_VAL" ] || [ -z "$MAX_COMP_VAL" ] ||
   [ -z "$AVG_COMM_VAL" ] || [ -z "$MAX_COMM_VAL" ] ||
   [ -z "$ITERS_VAL" ] || [ -z "$RESIDUAL_VAL" ]; then
  cat "$RUN_LOG"
  echo "Error: pilot log is missing one or more expected timing/validation fields." >&2
  exit 1
fi

CSV_FILE="$RESULTS_DIR/all_scaling_pilot_results.csv"
echo "N,steps,mpi_ranks,run,execution_time_s,avg_rank_comp_time_s,max_rank_comp_time_s,avg_rank_comm_time_s,max_rank_comm_time_s,total_jacobi_iters,max_residual" > "$CSV_FILE"
echo "$N,$STEPS,$P,1,$TIME_VAL,$AVG_COMP_VAL,$MAX_COMP_VAL,$AVG_COMM_VAL,$MAX_COMM_VAL,$ITERS_VAL,$RESIDUAL_VAL" >> "$CSV_FILE"

cat > "$RESULTS_DIR/summary.txt" <<EOF
Strong-scaling pilot summary
Grid: N=$N, steps=$STEPS
MPI ranks: $P
Repetitions: 1
Execution time: $TIME_VAL seconds
Average-rank computation time: $AVG_COMP_VAL seconds
Maximum-rank computation time: $MAX_COMP_VAL seconds
Average-rank communication time: $AVG_COMM_VAL seconds
Maximum-rank communication time: $MAX_COMM_VAL seconds
Total Jacobi iterations: $ITERS_VAL
Maximum residual: $RESIDUAL_VAL
Run log: $RUN_LOG

Timing note: computation and communication timers cover only the regions instrumented in the C++ source;
they do not necessarily sum to the full execution time.
EOF

cat "$RESULTS_DIR/summary.txt"
