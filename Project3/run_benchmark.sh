#!/usr/bin/env bash
# ==============================================================================
# HPC Benchmark & Experiment Runner (Atlas / Linux / macOS)
# ==============================================================================
# This script:
#   1. Compiles the target C++ program with optimal flags (-O3, MPI, OpenMP).
#   2. Runs the experiment K times (default: 5).
#   3. Captures raw output logs for every run into results/logs/.
#   4. Extracts "Execution time" from each run and computes:
#        - Mean, Min, Max, and Standard Deviation.
#   5. Appends summary data to results/benchmark_summary.txt and results/all_results.csv.
#
# Usage:
#   ./run_benchmark.sh <file.cpp> <mpi_procs> <omp_threads> [runs=5] [args...]
#
# Examples:
#   # Run serial implicit 5 times:
#   ./run_benchmark.sh heat_serial_implicit.cpp 1 1 5
#
#   # Run MPI implicit with 4 processes, 5 times:
#   ./run_benchmark.sh heat_mpi_implicit.cpp 4 1 5
#
#   # Run Hybrid MPI+OpenMP with 2 ranks and 4 threads, 5 times:
#   ./run_benchmark.sh heat_mpi_openMP_implicit.cpp 2 4 5
#
#   # Run with custom grid size (N=500, STEPS=100):
#   ./run_benchmark.sh heat_mpi_implicit.cpp 4 1 5 500 100
# ==============================================================================

set -euo pipefail

# 1. Parse Arguments
SOURCE_FILE="${1:-}"
MPI_PROCS="${2:-1}"
OMP_THREADS="${3:-1}"
NUM_RUNS="${4:-5}"
shift 4 2>/dev/null || shift $# 2>/dev/null || true
EXTRA_ARGS="$*"

if [ -z "$SOURCE_FILE" ] || [ ! -f "$SOURCE_FILE" ]; then
  echo "Usage: $0 <source_file.cpp> <mpi_procs> <omp_threads> [num_runs=5] [extra_args...]"
  echo "Error: Source file '$SOURCE_FILE' not found."
  exit 1
fi

BASE_NAME="$(basename "$SOURCE_FILE" .cpp)"
BINARY="./bin/${BASE_NAME}"
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
EXP_ID="${BASE_NAME}_p${MPI_PROCS}_t${OMP_THREADS}"

RESULTS_DIR="./results"
LOGS_DIR="${RESULTS_DIR}/logs"
SUMMARY_FILE="${RESULTS_DIR}/${EXP_ID}_summary.txt"
CSV_FILE="${RESULTS_DIR}/all_results.csv"

mkdir -p "./bin" "$RESULTS_DIR" "$LOGS_DIR"

# Ensure Atlas cluster GCC libraries (MPFR, GMP, MPC) are present if running on Atlas:
if [ -d "/app1/centos6.3/gnu/mpfr-2.4.2/lib" ]; then
  export LD_LIBRARY_PATH="/app1/centos6.3/gnu/mpfr-2.4.2/lib:/app1/centos6.3/gnu/gmp-4.3.2/lib:/app1/centos6.3/gnu/mpc-0.8.1/lib:${LD_LIBRARY_PATH:-}"
fi

# 2. Detect Compiler & Hardware Details (Platform Aware: Linux / Atlas vs macOS)
OS="$(uname -s)"
NODE_HOST="$(hostname)"

if [ "$OS" = "Darwin" ]; then
  CPU_MODEL="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
  CPU_CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo 'Unknown')"
  TOTAL_RAM="$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.2f GB\n", $1/1024/1024/1024}' || echo 'Unknown')"
  CACHE_INFO="$(sysctl -n hw.l3cachesize 2>/dev/null | awk '{if ($1>0) printf "%.0f KB (L3)\n", $1/1024}' || sysctl -n hw.l2cachesize 2>/dev/null | awk '{if ($1>0) printf "%.0f KB (L2)\n", $1/1024}' || echo 'N/A')"
else
  CPU_MODEL="$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || lscpu 2>/dev/null | grep "Model name" | cut -d: -f2 | xargs || echo 'Unknown CPU')"
  CPU_CORES="$(nproc 2>/dev/null || grep -c '^processor' /proc/cpuinfo 2>/dev/null || echo 'Unknown')"
  TOTAL_RAM="$(free -h 2>/dev/null | awk '/^Mem:/ {print $2}' || awk '/MemTotal/ {printf "%.2f GB\n", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 'Unknown')"
  CACHE_INFO="$(lscpu 2>/dev/null | grep "L3 cache" | cut -d: -f2 | xargs || echo 'N/A')"
fi
[ -z "$CACHE_INFO" ] && CACHE_INFO="N/A"

NEEDS_MPI=false
NEEDS_OMP=false

if grep -q "<mpi.h>" "$SOURCE_FILE"; then
  NEEDS_MPI=true
fi
if grep -q "<omp.h>" "$SOURCE_FILE" || grep -q "#pragma omp" "$SOURCE_FILE"; then
  NEEDS_OMP=true
fi

echo "=========================================================="
echo "  Compute Node Hardware Details"
echo "=========================================================="
echo "Node Hostname   : $NODE_HOST"
echo "CPU Model       : $CPU_MODEL"
echo "Total CPU Cores : $CPU_CORES"
echo "Total RAM       : $TOTAL_RAM"
echo "Cache Info      : $CACHE_INFO"
echo "=========================================================="
echo "Target Source   : $SOURCE_FILE"
echo "Experiment ID   : $EXP_ID"
echo "MPI Processes   : $MPI_PROCS"
echo "OpenMP Threads  : $OMP_THREADS"
echo "Repetitions (K) : $NUM_RUNS"
echo "Extra Arguments : ${EXTRA_ARGS:-'(default parameters)'}"
echo "Platform OS     : $OS"
echo "----------------------------------------------------------"

# Compiler selection
if [ "$NEEDS_MPI" = true ]; then
  CXX="${CXX:-mpicxx}"
else
  CXX="${CXX:-g++}"
fi

CXXFLAGS="-O3 -std=c++17"

if [ "$NEEDS_OMP" = true ]; then
  if [ "$OS" = "Darwin" ]; then
    # macOS with Homebrew libomp
    BREW_LIBOMP="/opt/homebrew/opt/libomp"
    if [ -d "$BREW_LIBOMP" ]; then
      CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -I${BREW_LIBOMP}/include -L${BREW_LIBOMP}/lib -lomp"
    else
      CXXFLAGS="$CXXFLAGS -fopenmp"
    fi
  else
    # Linux / Atlas Cluster standard GCC
    CXXFLAGS="$CXXFLAGS -fopenmp"
  fi
fi

# 3. Compile
echo "[1/3] Compiling: $CXX $CXXFLAGS $SOURCE_FILE -o $BINARY"
$CXX $CXXFLAGS "$SOURCE_FILE" -o "$BINARY"
echo "      Compilation successful."
echo "----------------------------------------------------------"

# 4. Prepare MPI runner command
RUN_CMD=()
if [ "$NEEDS_MPI" = true ]; then
  # Check if mpirun or srun is preferred
  if command -v srun >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
    RUN_CMD=(srun -n "$MPI_PROCS")
  elif command -v mpirun >/dev/null 2>&1; then
    # On macOS or unmanaged environments, allow oversubscribe if needed
    if [ "$OS" = "Darwin" ]; then
      RUN_CMD=(mpirun --oversubscribe --bind-to none -np "$MPI_PROCS")
    else
      RUN_CMD=(mpirun -np "$MPI_PROCS")
    fi
  elif command -v mpiexec >/dev/null 2>&1; then
    RUN_CMD=(mpiexec -n "$MPI_PROCS")
  else
    echo "Warning: No MPI launcher found (mpirun/srun), running directly."
    RUN_CMD=()
  fi
fi

# 5. Execute Repetitions
echo "[2/3] Executing $NUM_RUNS runs..."
export OMP_NUM_THREADS="$OMP_THREADS"

RUN_TIMES=()
COMP_TIMES=()
COMM_TIMES=()

for ((k = 1; k <= NUM_RUNS; k++)); do
  LOG_FILE="${LOGS_DIR}/${EXP_ID}_run${k}_${TIMESTAMP}.log"
  echo -n "      -> Run $k / $NUM_RUNS ... "

  # Run and log output
  if [ ${#RUN_CMD[@]} -gt 0 ]; then
    "${RUN_CMD[@]}" "$BINARY" $EXTRA_ARGS > "$LOG_FILE" 2>&1
  else
    "$BINARY" $EXTRA_ARGS > "$LOG_FILE" 2>&1
  fi

  # Extract execution time (supports various formats printed by your codes)
  # Looks for: "Execution time : <float>" or "Total Execution Time: <float>"
  TIME_VAL=$(grep -i "Execution time" "$LOG_FILE" | head -n 1 | grep -oE '[0-9]+\.[0-9]+' | head -n 1 || true)

  if [ -z "$TIME_VAL" ]; then
    # Fallback to Elapsed time if present
    TIME_VAL=$(grep -i "elapsed" "$LOG_FILE" | head -n 1 | grep -oE '[0-9]+\.[0-9]+' | head -n 1 || true)
  fi

  # Extract computation and communication times if instrumented
  COMP_VAL=$(grep -i "Computation Time" "$LOG_FILE" | head -n 1 | grep -oE '[0-9]+\.[0-9]+' | head -n 1 || true)
  COMM_VAL=$(grep -i "Communication Time" "$LOG_FILE" | head -n 1 | grep -oE '[0-9]+\.[0-9]+' | head -n 1 || true)

  if [ -n "$TIME_VAL" ]; then
    RUN_TIMES+=("$TIME_VAL")
    [ -n "$COMP_VAL" ] && COMP_TIMES+=("$COMP_VAL")
    [ -n "$COMM_VAL" ] && COMM_TIMES+=("$COMM_VAL")
    echo "Done! (${TIME_VAL} s)"
  else
    echo "Done! (Time parsing fallback, see log: $LOG_FILE)"
  fi
done

echo "----------------------------------------------------------"

# 6. Compute Statistics (using awk for portable floating point math)
if [ ${#RUN_TIMES[@]} -eq 0 ]; then
  echo "Error: No valid execution times could be extracted from logs."
  exit 1
fi

STATS=$(awk -v k="${#RUN_TIMES[@]}" 'BEGIN {
  sum = 0; min = 999999999; max = -1;
  for (i = 1; i < ARGC; i++) {
    val = ARGV[i] + 0.0;
    sum += val;
    if (val < min) min = val;
    if (val > max) max = val;
  }
  mean = sum / k;
  sq_sum = 0;
  for (i = 1; i < ARGC; i++) {
    val = ARGV[i] + 0.0;
    sq_sum += (val - mean) * (val - mean);
  }
  stddev = (k > 1) ? sqrt(sq_sum / (k - 1)) : 0.0;
  printf "%.6f %.6f %.6f %.6f\n", mean, min, max, stddev;
}' "${RUN_TIMES[@]}")

MEAN_TIME=$(echo "$STATS" | awk '{print $1}')
MIN_TIME=$(echo "$STATS" | awk '{print $2}')
MAX_TIME=$(echo "$STATS" | awk '{print $3}')
STDDEV_TIME=$(echo "$STATS" | awk '{print $4}')

# Compute average computation and communication times if available
MEAN_COMP_TIME="0.000000"
if [ ${#COMP_TIMES[@]} -gt 0 ]; then
  MEAN_COMP_TIME=$(awk -v k="${#COMP_TIMES[@]}" 'BEGIN {
    sum = 0;
    for (i = 1; i < ARGC; i++) sum += ARGV[i] + 0.0;
    printf "%.6f\n", sum / k;
  }' "${COMP_TIMES[@]}")
fi

MEAN_COMM_TIME="0.000000"
if [ ${#COMM_TIMES[@]} -gt 0 ]; then
  MEAN_COMM_TIME=$(awk -v k="${#COMM_TIMES[@]}" 'BEGIN {
    sum = 0;
    for (i = 1; i < ARGC; i++) sum += ARGV[i] + 0.0;
    printf "%.6f\n", sum / k;
  }' "${COMM_TIMES[@]}")
fi

# Track serial baseline time to compute Speedup and Parallel Efficiency
SERIAL_TIME_FILE="${RESULTS_DIR}/serial_baseline_time.txt"
SPEEDUP=""
EFFICIENCY=""

if [[ "$SOURCE_FILE" == *"serial"* ]]; then
  echo "$MEAN_TIME" > "$SERIAL_TIME_FILE"
  SPEEDUP="1.00"
  EFFICIENCY="100.00%"
elif [ -f "$SERIAL_TIME_FILE" ]; then
  SERIAL_TIME=$(cat "$SERIAL_TIME_FILE" | tr -d '[:space:]')
  if [ -n "$SERIAL_TIME" ]; then
    SPEEDUP=$(awk -v s="$SERIAL_TIME" -v m="$MEAN_TIME" 'BEGIN {
      if (m > 0) printf "%.2f", s / m; else print "N/A";
    }')
    EFFICIENCY=$(awk -v s="$SERIAL_TIME" -v m="$MEAN_TIME" -v p="$MPI_PROCS" 'BEGIN {
      if (m > 0 && p > 0) printf "%.2f%%", (s / (m * p)) * 100.0; else print "N/A";
    }')
  fi
fi

# 7. Write Human-Readable Labeled Summary File
{
  echo "================================================================================"
  echo "Experiment Summary: ${EXP_ID}"
  echo "Timestamp          : $(date)"
  echo "Node Hostname      : ${NODE_HOST}"
  echo "CPU Model          : ${CPU_MODEL}"
  echo "Total CPU Cores    : ${CPU_CORES}"
  echo "Total RAM          : ${TOTAL_RAM}"
  echo "Cache Info         : ${CACHE_INFO}"
  echo "================================================================================"
  echo "Source Code        : ${SOURCE_FILE}"
  echo "Binary             : ${BINARY}"
  echo "MPI Processes      : ${MPI_PROCS}"
  echo "OpenMP Threads     : ${OMP_THREADS}"
  echo "Repetitions        : ${NUM_RUNS}"
  echo "Arguments Passed   : ${EXTRA_ARGS:-'(default)'}"
  echo "--------------------------------------------------------------------------------"
  echo "Individual Run Times (seconds):"
  for ((i = 0; i < ${#RUN_TIMES[@]}; i++)); do
    idx=$((i + 1))
    printf "  Run %2d : %10.6f s\n" "$idx" "${RUN_TIMES[$i]}"
  done
  echo "--------------------------------------------------------------------------------"
  printf "Mean Execution Time : %10.6f s\n" "$MEAN_TIME"
  printf "  - Mean Comp Time  : %10.6f s\n" "$MEAN_COMP_TIME"
  printf "  - Mean Comm Time  : %10.6f s\n" "$MEAN_COMM_TIME"
  printf "Min (Best) Time     : %10.6f s\n" "$MIN_TIME"
  printf "Max (Worst) Time    : %10.6f s\n" "$MAX_TIME"
  printf "Sample Std. Dev.    : %10.6f s\n" "$STDDEV_TIME"
  if [ -n "$SPEEDUP" ]; then
    printf "Parallel Speedup    : %10s\n" "${SPEEDUP}x"
    printf "Parallel Efficiency : %10s\n" "$EFFICIENCY"
  fi
  echo "================================================================================"
  echo ""
  echo "Detailed output from Run 1:"
  echo "--------------------------------------------------------------------------------"
  cat "${LOGS_DIR}/${EXP_ID}_run1_${TIMESTAMP}.log"
} > "$SUMMARY_FILE"

# 8. Append to Master CSV File for Plotting
if [ ! -f "$CSV_FILE" ]; then
  echo "timestamp,experiment,source_file,mpi_procs,omp_threads,num_runs,mean_time_s,speedup,efficiency,mean_comp_time_s,mean_comm_time_s,min_time_s,max_time_s,stddev_s,all_runs" > "$CSV_FILE"
fi

ALL_RUNS_STR=$(IFS=';'; echo "${RUN_TIMES[*]}")
echo "${TIMESTAMP},${EXP_ID},${SOURCE_FILE},${MPI_PROCS},${OMP_THREADS},${NUM_RUNS},${MEAN_TIME},${SPEEDUP:-N/A},${EFFICIENCY:-N/A},${MEAN_COMP_TIME},${MEAN_COMM_TIME},${MIN_TIME},${MAX_TIME},${STDDEV_TIME},\"${ALL_RUNS_STR}\"" >> "$CSV_FILE"

# 9. Console Output
echo "[3/3] Results Successfully Saved!"
echo "  -> Summary Text : $SUMMARY_FILE"
echo "  -> Master CSV   : $CSV_FILE"
echo "  -> Raw Logs     : ${LOGS_DIR}/"
echo "----------------------------------------------------------"
echo "  => MEAN TIME    : ${MEAN_TIME} s (Min: ${MIN_TIME}, Max: ${MAX_TIME}, StdDev: ${STDDEV_TIME})"
if [ -n "$SPEEDUP" ]; then
  echo "  => SPEEDUP      : ${SPEEDUP}x | EFFICIENCY: ${EFFICIENCY}"
fi
echo "  => COMPUTATION  : ${MEAN_COMP_TIME} s | COMMUNICATION: ${MEAN_COMM_TIME} s"
echo "=========================================================="

