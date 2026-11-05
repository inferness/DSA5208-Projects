#!/usr/bin/env bash
# Build and benchmark the single-GPU OpenMP target implementation.
# Usage: ./run_gpu_offload.sh [runs=3] [N=3000] [steps=1000] [lambda=0.05] [max_iters=25] [tol=1e-5]

set -euo pipefail

RUNS="${1:-3}"
N="${2:-3000}"
STEPS="${3:-1000}"
LAMBDA="${4:-0.05}"
MAX_ITERS="${5:-25}"
TOL="${6:-1e-5}"
GPU_CXX="${GPU_CXX:-nvc++}"
CPU_CXX="${CPU_CXX:-g++}"
GPU_OMP_THREADS="${GPU_OMP_THREADS:-8}"
GPU_CXXFLAGS="${GPU_CXXFLAGS:--O3 -std=c++17 -mp=gpu -Minfo=mp}"
GPU_RESULTS_DIR="${GPU_RESULTS_DIR:-gpu_offload_results}"

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

mkdir -p bin results/logs "$GPU_RESULTS_DIR/logs"
export GPU_RESULTS_DIR
read -r -a GPU_FLAGS <<< "$GPU_CXXFLAGS"

echo "GPU compiler: $GPU_CXX"
echo "GPU flags   : $GPU_CXXFLAGS"
echo "Problem     : N=$N STEPS=$STEPS RUNS=$RUNS"
echo "OpenMP host threads: $GPU_OMP_THREADS"

{
  echo "GPU offload job started: $(date)"
  echo "Node: $(hostname)"
  echo "GPU compiler: $GPU_CXX"
  "$GPU_CXX" --version 2>&1 || true
  echo "CPU reference compiler: $CPU_CXX"
  "$CPU_CXX" --version 2>&1 | head -n 2 || true
  echo "GPU inventory:"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>&1 || true
    nvidia-smi 2>&1 || true
  else
    echo "nvidia-smi not found"
  fi
  echo "Loaded modules:"
  module list 2>&1 || true
} | tee "$GPU_RESULTS_DIR/gpu_offload_toolchain.txt"

echo "[1/3] Compiling optimized CPU reference..."
"$CPU_CXX" -O3 -std=c++17 heat_serial_implicit_optimized.cpp \
  -o bin/heat_serial_implicit_optimized
CPU_REFERENCE_LOG="$GPU_RESULTS_DIR/logs/heat_serial_implicit_optimized_reference.log"
OMP_NUM_THREADS=1 ./bin/heat_serial_implicit_optimized \
  "$N" "$STEPS" "$LAMBDA" "$MAX_ITERS" "$TOL" > "$CPU_REFERENCE_LOG" 2>&1
echo "      CPU reference saved in $CPU_REFERENCE_LOG"

echo "[2/3] Compiling OpenMP GPU offload implementation..."
echo "      Command: $GPU_CXX $GPU_CXXFLAGS heat_openmp_gpu_offload.cpp -o bin/heat_openmp_gpu_offload"
"$GPU_CXX" "${GPU_FLAGS[@]}" heat_openmp_gpu_offload.cpp \
  -o bin/heat_openmp_gpu_offload 2>&1 | tee "$GPU_RESULTS_DIR/gpu_offload_compile.log"

RUN_TIMES=()
for ((run = 1; run <= RUNS; run++)); do
  LOG_FILE="$GPU_RESULTS_DIR/logs/heat_openmp_gpu_offload_run${run}_$(date +'%Y%m%d_%H%M%S').log"
  echo "[3/3] GPU run $run / $RUNS ..."
  OMP_NUM_THREADS="$GPU_OMP_THREADS" OMP_TARGET_OFFLOAD=MANDATORY \
    ./bin/heat_openmp_gpu_offload "$N" "$STEPS" "$LAMBDA" \
      "$MAX_ITERS" "$TOL" > "$LOG_FILE" 2>&1
  cat "$LOG_FILE"
  TIME_VAL="$(grep -i '^Execution time' "$LOG_FILE" | head -n 1 | grep -oE '[0-9]+\.[0-9]+' | head -n 1 || true)"
  if [ -z "$TIME_VAL" ]; then
    echo "Error: could not parse execution time from $LOG_FILE" >&2
    exit 1
  fi
  RUN_TIMES+=("$TIME_VAL")
done

STATS="$(awk -v n="${#RUN_TIMES[@]}" '{ values[NR]=$1; sum+=$1 }
  END {
    mean=sum/n; min=values[1]; max=values[1]; ss=0;
    for (i=1; i<=n; i++) {
      if (values[i]<min) min=values[i];
      if (values[i]>max) max=values[i];
      d=values[i]-mean; ss+=d*d;
    }
    sd=(n>1) ? sqrt(ss/(n-1)) : 0;
    printf "%.6f %.6f %.6f %.6f\n", mean,min,max,sd;
  }' < <(printf '%s\n' "${RUN_TIMES[@]}") )"
read -r MEAN_TIME MIN_TIME MAX_TIME STDDEV_TIME <<< "$STATS"

SUMMARY_FILE="$GPU_RESULTS_DIR/heat_openmp_gpu_offload_summary.txt"
{
  echo "=========================================================="
  echo "OpenMP GPU Offload Summary"
  echo "Source       : heat_openmp_gpu_offload.cpp"
  echo "GPU compiler : $GPU_CXX"
  echo "GPU flags    : $GPU_CXXFLAGS"
  echo "Grid/steps   : N=$N STEPS=$STEPS"
  echo "Runs         : $RUNS"
  echo "Target mode  : OpenMP target, GPU required"
  echo "----------------------------------------------------------"
  for ((i = 0; i < ${#RUN_TIMES[@]}; i++)); do
    printf "Run %d        : %s s\n" "$((i + 1))" "${RUN_TIMES[$i]}"
  done
  echo "Mean         : $MEAN_TIME s"
  echo "Min          : $MIN_TIME s"
  echo "Max          : $MAX_TIME s"
  echo "Std. dev.    : $STDDEV_TIME s"
  echo "CPU reference: $CPU_REFERENCE_LOG"
  echo "Toolchain    : $GPU_RESULTS_DIR/gpu_offload_toolchain.txt"
  echo "=========================================================="
} | tee "$SUMMARY_FILE"
