#!/usr/bin/env bash
# Compare optimized and original serial implementations over multiple grid sizes.
# Usage: ./run_serial_size_sweep.sh [steps=1000] [lambda=0.05] [max_iters=25] [tol=1e-5]

set -euo pipefail

STEPS="${1:-1000}"
LAMBDA="${2:-0.05}"
MAX_ITERS="${3:-25}"
TOL="${4:-1e-5}"
GRID_SIZES=(1024 2048 4096 8192 16384)
REPEATS_SMALL="${SERIAL_RUNS:-5}"
RESULTS_ROOT="${RESULTS_DIR:-results/serial_size_sweep}"
BIN_DIR="${BIN_DIR:-bin/serial_size_sweep}"
CXX="${CXX:-mpicxx}"
CXXFLAGS="${CXXFLAGS:--O3 -std=c++17}"
SOURCES=(heat_serial_implicit_optimized.cpp heat_serial_implicit_unoptimized.cpp)
VARIANTS=(optimized unoptimized)
RUN_CMD=()
if command -v mpirun >/dev/null 2>&1; then
  RUN_CMD=(mpirun -np 1 --bind-to core)
fi

if ! [[ "$STEPS" =~ ^[0-9]+$ && "$MAX_ITERS" =~ ^[1-9][0-9]*$ && "$REPEATS_SMALL" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: steps must be nonnegative; max iterations and SERIAL_RUNS must be positive integers." >&2
  exit 2
fi
if ! command -v "$CXX" >/dev/null 2>&1; then
  echo "Error: compiler '$CXX' not found. Load Vanda's GCC/OpenMPI modules." >&2
  exit 2
fi

mkdir -p "$RESULTS_ROOT/logs" "$RESULTS_ROOT/artifacts" "$BIN_DIR"
read -r -a FLAGS <<< "$CXXFLAGS"

{
  echo "Serial grid-size sweep started: $(date)"
  echo "Node: $(hostname)"
  echo "Compiler: $CXX"
  "$CXX" --version 2>&1 | head -n 2 || true
  echo "Compiler flags: $CXXFLAGS"
  echo "Grid sizes: ${GRID_SIZES[*]}"
  echo "Steps=$STEPS lambda=$LAMBDA max_iters=$MAX_ITERS tolerance=$TOL"
  echo "Repetitions: $REPEATS_SMALL for N <= 8192; 1 for N=16384"
  echo "Variants: optimized and unoptimized serial"
} | tee "$RESULTS_ROOT/serial_size_sweep_toolchain.txt"

echo "Compiling serial variants..."
for i in "${!SOURCES[@]}"; do
  "$CXX" "${FLAGS[@]}" "${SOURCES[$i]}" -o "$BIN_DIR/heat_serial_${VARIANTS[$i]}"
done

RAW_CSV="$RESULTS_ROOT/all_serial_size_runs.csv"
echo "N,steps,variant,run,execution_time_s,total_jacobi_iters,max_residual" > "$RAW_CSV"

for N in "${GRID_SIZES[@]}"; do
  RUNS="$REPEATS_SMALL"
  if [ "$N" -eq 16384 ]; then RUNS=1; fi

  for i in "${!VARIANTS[@]}"; do
    VARIANT="${VARIANTS[$i]}"
    BINARY="$BIN_DIR/heat_serial_${VARIANT}"
    for ((run=1; run<=RUNS; run++)); do
      LOG_FILE="$RESULTS_ROOT/logs/N${N}_${VARIANT}_run${run}_$(date +'%Y%m%d_%H%M%S').log"
      ARTIFACT_DIR="$RESULTS_ROOT/artifacts/N${N}_${VARIANT}_run${run}"
      mkdir -p "$ARTIFACT_DIR"
      echo "Running N=$N variant=$VARIANT run=$run/$RUNS"

      RESULTS_DIR="$ARTIFACT_DIR" "${RUN_CMD[@]}" "$BINARY" \
        "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" > "$LOG_FILE" 2>&1

      TIME_VAL="$(awk -F: '/^Execution time[[:space:]]*:/ {gsub(/ seconds/, "", $2); gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
      ITER_VAL="$(awk -F: '/^Total Jacobi iters[[:space:]]*:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
      RESIDUAL_VAL="$(awk -F: '/^Max absolute residual[[:space:]]*:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$LOG_FILE")"
      if [ -z "$TIME_VAL" ] || [ -z "$ITER_VAL" ] || [ -z "$RESIDUAL_VAL" ]; then
        cat "$LOG_FILE"
        echo "Error: missing timing or solver metrics in $LOG_FILE" >&2
        exit 1
      fi

      echo "$N,$STEPS,$VARIANT,$run,$TIME_VAL,$ITER_VAL,$RESIDUAL_VAL" >> "$RAW_CSV"
      # Discard the large saved grid after the run; retain compact metric text and logs.
      rm -f -- "$ARTIFACT_DIR/baseline_implicit_N${N}_S${STEPS}.bin"
    done
  done
done

AGG_CSV="$RESULTS_ROOT/all_serial_size_aggregated.csv"
python3 - "$RAW_CSV" "$AGG_CSV" <<'PY'
import csv
import math
import sys
from collections import defaultdict

source, destination = sys.argv[1:]
groups = defaultdict(list)
with open(source, newline='') as f:
    for row in csv.DictReader(f):
        groups[(row['N'], row['steps'], row['variant'])].append(row)

fields = ['N', 'steps', 'variant', 'runs', 'mean_time_s', 'stddev_time_s',
          'min_time_s', 'max_time_s', 'mean_jacobi_iters', 'max_residual']
with open(destination, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for (n, steps, variant), rows in sorted(groups.items(), key=lambda x: (int(x[0][0]), x[0][2])):
        times = [float(r['execution_time_s']) for r in rows]
        iters = [float(r['total_jacobi_iters']) for r in rows]
        mean = sum(times) / len(times)
        sd = math.sqrt(sum((t - mean) ** 2 for t in times) / (len(times) - 1)) if len(times) > 1 else 0.0
        writer.writerow({
            'N': n, 'steps': steps, 'variant': variant, 'runs': len(rows),
            'mean_time_s': f'{mean:.6f}', 'stddev_time_s': f'{sd:.6f}',
            'min_time_s': f'{min(times):.6f}', 'max_time_s': f'{max(times):.6f}',
            'mean_jacobi_iters': f'{sum(iters) / len(iters):.3f}',
            'max_residual': f"{max(float(r['max_residual']) for r in rows):.9e}",
        })
PY

echo "Serial size sweep complete."
echo "Per-run CSV: $RAW_CSV"
echo "Aggregated CSV: $AGG_CSV"
echo "Logs: $RESULTS_ROOT/logs/"
