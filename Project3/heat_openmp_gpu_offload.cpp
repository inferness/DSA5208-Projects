#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <set>
#include <string>
#include <vector>
#include "result_paths.hpp"

using namespace std;

#ifndef GPU_ROW_ONLY
#define GPU_ROW_ONLY 0
#endif
#ifndef GPU_MAPPING_POLICY
#define GPU_MAPPING_POLICY 0
#endif

inline size_t idx(size_t i, size_t j, size_t n) { return i * n + j; }

int main(int argc, char *argv[]) {
  int N = 3000;
  int STEPS = 1000;
  double LAMBDA = 0.05;
  int MAX_JACOBI_ITERS = 25;
  double TOL = 1e-5;

  if (argc >= 2) N = atoi(argv[1]);
  if (argc >= 3) STEPS = atoi(argv[2]);
  if (argc >= 4) LAMBDA = atof(argv[3]);
  if (argc >= 5) MAX_JACOBI_ITERS = atoi(argv[4]);
  if (argc >= 6) TOL = atof(argv[5]);

  if (N < 3 || STEPS < 0 || LAMBDA <= 0.0 || MAX_JACOBI_ITERS < 1 ||
      TOL <= 0.0) {
    cerr << "Error: Invalid arguments. N >= 3, STEPS >= 0, LAMBDA > 0, "
            "MAX_JACOBI_ITERS >= 1, TOL > 0.\n";
    return 1;
  }

  const int deviceCount = omp_get_num_devices();
  if (deviceCount < 1) {
    cerr << "Error: OpenMP reports no GPU target devices. Refusing to run on "
            "the CPU.\n";
    return 2;
  }
  omp_set_default_device(0);
  int ranOnInitialDevice = 1;
#pragma omp target map(from : ranOnInitialDevice)
  { ranOnInitialDevice = omp_is_initial_device(); }
  if (ranOnInitialDevice) {
    cerr << "Error: OpenMP target region ran on the host instead of a GPU.\n";
    return 2;
  }

  const double diag = 1.0 + 4.0 * LAMBDA;
  const double invDiag = 1.0 / diag;
  const size_t cellCount = static_cast<size_t>(N) * static_cast<size_t>(N);
  vector<double> uPrev(cellCount, 0.0);
  vector<double> gridA(cellCount, 0.0);
  vector<double> gridB(cellCount, 0.0);

  const int center = N / 2;
  uPrev[idx(center, center, N)] = 100.0;
  gridA[idx(center, center, N)] = 100.0;
  gridB[idx(center, center, N)] = 100.0;

  cout << "========================================================\n";
  cout << "  2D Heat Equation - OpenMP GPU Offload (No MPI)       \n";
  cout << "  Method: Backward Euler + Jacobi Iterative Solver     \n";
  cout << "========================================================\n";
  cout << "Grid size (N x N)    : " << N << " x " << N << "\n";
  cout << "Time steps           : " << STEPS << "\n";
  cout << "Lambda (diffusion)   : " << LAMBDA << "\n";
  cout << "Max Jacobi Iters/Step: " << MAX_JACOBI_ITERS << "\n";
  cout << "Convergence Tol      : " << TOL << "\n";
  cout << "OpenMP target devices: " << deviceCount << "\n";
  cout << "Selected target      : device 0 (GPU confirmed)\n";
#if GPU_MAPPING_POLICY == 0
  cout << "Data mapping         : persistent for full simulation\n";
#elif GPU_MAPPING_POLICY == 1
  cout << "Data mapping         : mapped once per physical time step\n";
#else
  cout << "Data mapping         : mapped for each Jacobi iteration\n";
#endif
#if GPU_ROW_ONLY
  cout << "Loop mapping         : row-only (no collapse)\n";
#else
  cout << "Loop mapping         : cell-level collapse(2)\n";
#endif
  cout << "Running simulation...\n";

  double *rhs = uPrev.data();
  double *a = gridA.data();
  double *b = gridB.data();
  bool currentIsA = true;
  long long totalJacobiIters = 0;
  int nonConvergedSteps = 0;
  double maxResidualOverall = 0.0;
  int maxTeamsObserved = 0;
  int minTeamsObserved = numeric_limits<int>::max();
  long long sumTeamsObserved = 0;
  long long teamLaunchCount = 0;
  set<int> observedTeamCounts;

  const auto start = chrono::steady_clock::now();
  {
#if GPU_MAPPING_POLICY == 0
    // Keep all three grids resident on the GPU for the complete solve. The
    // nested target regions map already-present sections, so each iteration
    // transfers only the scalar convergence reduction.
#pragma omp target data map(tofrom : rhs[0 : cellCount], a[0 : cellCount], b[0 : cellCount])
#endif
    {
      for (int t = 0; t < STEPS; ++t) {
#if GPU_MAPPING_POLICY == 1
        // Recreate device mappings at each physical time step. The Jacobi
        // iterations for this step reuse these device allocations.
#pragma omp target data map(tofrom : rhs[0 : cellCount], a[0 : cellCount], b[0 : cellCount])
#endif
        {
        bool converged = false;
        int iter = 0;

        for (; iter < MAX_JACOBI_ITERS; ++iter) {
          double maxDiff = 0.0;
          int teamsThisLaunch = 0;

          if (currentIsA) {
#if GPU_ROW_ONLY
#pragma omp target teams distribute parallel for reduction(max : maxDiff, teamsThisLaunch) map(to : rhs[0 : cellCount], a[0 : cellCount]) map(from : b[0 : cellCount])
#else
#pragma omp target teams distribute parallel for collapse(2) reduction(max : maxDiff, teamsThisLaunch) map(to : rhs[0 : cellCount], a[0 : cellCount]) map(from : b[0 : cellCount])
#endif
            for (int i = 1; i < N - 1; ++i) {
              for (int j = 1; j < N - 1; ++j) {
                if (j == 1) teamsThisLaunch = omp_get_num_teams();
                const size_t k = static_cast<size_t>(i) * N + j;
                const double neighbors = a[k - N] + a[k + N] + a[k - 1] + a[k + 1];
                const double newValue = (rhs[k] + LAMBDA * neighbors) * invDiag;
                b[k] = newValue;
                const double diff = fabs(newValue - a[k]);
                if (diff > maxDiff) maxDiff = diff;
              }
            }
            currentIsA = false;
          } else {
#if GPU_ROW_ONLY
#pragma omp target teams distribute parallel for reduction(max : maxDiff, teamsThisLaunch) map(to : rhs[0 : cellCount], b[0 : cellCount]) map(from : a[0 : cellCount])
#else
#pragma omp target teams distribute parallel for collapse(2) reduction(max : maxDiff, teamsThisLaunch) map(to : rhs[0 : cellCount], b[0 : cellCount]) map(from : a[0 : cellCount])
#endif
            for (int i = 1; i < N - 1; ++i) {
              for (int j = 1; j < N - 1; ++j) {
                if (j == 1) teamsThisLaunch = omp_get_num_teams();
                const size_t k = static_cast<size_t>(i) * N + j;
                const double neighbors = b[k - N] + b[k + N] + b[k - 1] + b[k + 1];
                const double newValue = (rhs[k] + LAMBDA * neighbors) * invDiag;
                a[k] = newValue;
                const double diff = fabs(newValue - b[k]);
                if (diff > maxDiff) maxDiff = diff;
              }
            }
            currentIsA = true;
          }

          // Sample the runtime-selected number of teams for every Jacobi
          // kernel launch, so automatic team selection can be compared both
          // within a run and across repetitions.
          if (teamsThisLaunch > 0) {
            minTeamsObserved = min(minTeamsObserved, teamsThisLaunch);
            maxTeamsObserved = max(maxTeamsObserved, teamsThisLaunch);
            sumTeamsObserved += teamsThisLaunch;
            ++teamLaunchCount;
            observedTeamCounts.insert(teamsThisLaunch);
          }

          if (maxDiff < TOL) {
            converged = true;
            ++iter;
            break;
          }
        }

        if (!converged) {
          ++nonConvergedSteps;
          if (nonConvergedSteps <= 5) {
            cerr << "Warning: Jacobi did not converge at time step " << t
                 << " (reached max " << MAX_JACOBI_ITERS << " iterations).\n";
          } else if (nonConvergedSteps == 6) {
            cerr << "Warning: Further convergence warnings suppressed...\n";
          }
        }
        totalJacobiIters += iter;

        // Compute the final residual before replacing the right-hand side.
        if (t == STEPS - 1) {
          double stepMaxResidual = 0.0;
          if (currentIsA) {
#if GPU_ROW_ONLY
#pragma omp target teams distribute parallel for reduction(max : stepMaxResidual) map(to : rhs[0 : cellCount], a[0 : cellCount])
#else
#pragma omp target teams distribute parallel for collapse(2) reduction(max : stepMaxResidual) map(to : rhs[0 : cellCount], a[0 : cellCount])
#endif
            for (int i = 1; i < N - 1; ++i) {
              for (int j = 1; j < N - 1; ++j) {
                const size_t k = static_cast<size_t>(i) * N + j;
                const double au = diag * a[k] -
                                  LAMBDA * (a[k - N] + a[k + N] + a[k - 1] + a[k + 1]);
                const double residual = fabs(rhs[k] - au);
                if (residual > stepMaxResidual) stepMaxResidual = residual;
              }
            }
          } else {
#if GPU_ROW_ONLY
#pragma omp target teams distribute parallel for reduction(max : stepMaxResidual) map(to : rhs[0 : cellCount], b[0 : cellCount])
#else
#pragma omp target teams distribute parallel for collapse(2) reduction(max : stepMaxResidual) map(to : rhs[0 : cellCount], b[0 : cellCount])
#endif
            for (int i = 1; i < N - 1; ++i) {
              for (int j = 1; j < N - 1; ++j) {
                const size_t k = static_cast<size_t>(i) * N + j;
                const double au = diag * b[k] -
                                  LAMBDA * (b[k - N] + b[k + N] + b[k - 1] + b[k + 1]);
                const double residual = fabs(rhs[k] - au);
                if (residual > stepMaxResidual) stepMaxResidual = residual;
              }
            }
          }
          maxResidualOverall = stepMaxResidual;
        } else if (currentIsA) {
#pragma omp target teams distribute parallel for map(to : a[0 : cellCount]) map(from : rhs[0 : cellCount])
          for (size_t k = 0; k < cellCount; ++k) rhs[k] = a[k];
        } else {
#pragma omp target teams distribute parallel for map(to : b[0 : cellCount]) map(from : rhs[0 : cellCount])
          for (size_t k = 0; k < cellCount; ++k) rhs[k] = b[k];
        }
        }
      }
    }
  }
  const auto end = chrono::steady_clock::now();
  const double elapsed = chrono::duration<double>(end - start).count();

  const double *finalGrid = currentIsA ? gridA.data() : gridB.data();
  double maxTemperature = 0.0;
  double minTemperature = numeric_limits<double>::infinity();
  double totalHeat = 0.0;
  for (size_t k = 0; k < cellCount; ++k) {
    maxTemperature = max(maxTemperature, finalGrid[k]);
    minTemperature = min(minTemperature, finalGrid[k]);
    totalHeat += finalGrid[k];
  }

  const char *resultDirEnv = getenv("GPU_RESULTS_DIR");
  const string resultDir = resultDirEnv ? resultDirEnv : "gpu_offload_results";
  filesystem::create_directories(resultDir);
  const string gpuGridPath = resultDir + "/heat_openmp_gpu_offload_N" +
                             to_string(N) + "_S" + to_string(STEPS) + ".bin";
  bool saveGpuGrid = true;
  if (const char *saveGridEnv = getenv("GPU_SAVE_RESULT_GRID")) {
    saveGpuGrid = string(saveGridEnv) != "0";
  }
  if (saveGpuGrid) {
    ofstream out(gpuGridPath, ios::binary);
    if (!out || !out.write(reinterpret_cast<const char *>(finalGrid),
                           static_cast<streamsize>(cellCount * sizeof(double)))) {
      cerr << "Error: Could not write GPU result grid to " << gpuGridPath << ".\n";
      return 1;
    }
  }

  double maxBaselineDifference = -1.0;
  const string baselinePath = result_path("baseline_implicit_N" + to_string(N) +
                                          "_S" + to_string(STEPS) + ".bin");
  {
    ifstream baseline(baselinePath, ios::binary);
    if (baseline) {
      vector<double> baselineGrid(cellCount);
      if (baseline.read(reinterpret_cast<char *>(baselineGrid.data()),
                        static_cast<streamsize>(cellCount * sizeof(double)))) {
        maxBaselineDifference = 0.0;
        for (size_t k = 0; k < cellCount; ++k) {
          maxBaselineDifference =
              max(maxBaselineDifference, fabs(finalGrid[k] - baselineGrid[k]));
        }
      }
    }
  }

  cout << fixed << setprecision(6);
  cout << "========================================================\n";
  cout << "Execution time       : " << elapsed << " seconds\n";
  cout << "Min teams observed   : "
       << (teamLaunchCount > 0 ? minTeamsObserved : 0) << "\n";
  cout << "Max teams observed   : " << maxTeamsObserved << "\n";
  cout << "Mean teams per launch: "
       << (teamLaunchCount > 0
               ? static_cast<double>(sumTeamsObserved) / teamLaunchCount
               : 0.0)
       << "\n";
  cout << "Team launches sampled: " << teamLaunchCount << "\n";
  cout << "Observed team count values: ";
  if (observedTeamCounts.empty()) {
    cout << "none";
  } else {
    bool firstTeamCount = true;
    for (int teamCount : observedTeamCounts) {
      if (!firstTeamCount) cout << "|";
      cout << teamCount;
      firstTeamCount = false;
    }
  }
  cout << "\n";
  cout << "Avg Jacobi iters/step: "
       << (STEPS > 0 ? static_cast<double>(totalJacobiIters) / STEPS : 0.0)
       << "\n";
  cout << "Total Jacobi iters   : " << totalJacobiIters << "\n";
  cout << "Non-converged steps  : " << nonConvergedSteps << " / " << STEPS << "\n";
  cout << scientific << setprecision(6);
  cout << "Max absolute residual: " << maxResidualOverall << "\n";
  cout << fixed << setprecision(6);
  cout << "Center Temperature   : " << finalGrid[idx(center, center, N)] << "\n";
  cout << "Maximum Temperature  : " << maxTemperature << "\n";
  cout << "Minimum Temperature  : " << minTemperature << "\n";
  cout << "Total Heat Energy    : " << totalHeat << "\n";
  cout << "GPU result grid      : "
       << (saveGpuGrid ? gpuGridPath : "not saved (GPU_SAVE_RESULT_GRID=0)")
       << "\n";
  if (maxBaselineDifference >= 0.0) {
    cout << scientific << "Max difference vs CPU baseline: "
         << maxBaselineDifference << "\n";
    cout << "CPU baseline check   : "
         << (maxBaselineDifference <= 1e-8 ? "PASSED" : "WARNING") << "\n";
  } else {
    cout << "CPU baseline check   : unavailable (baseline grid not found)\n";
  }
  cout << "========================================================\n";
  return 0;
}
