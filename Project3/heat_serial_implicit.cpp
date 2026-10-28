#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#endif

using namespace std;

// Convert 2D grid coordinates (i, j) to flat 1D index
inline size_t idx(size_t i, size_t j, size_t n) {
  return i * n + j;
}

int main(int argc, char* argv[]) {
  // Default parameters matching heat_serial.cpp
  int N = 3000;
  int STEPS = 1000;
  // Discretization parameters:
  // LAMBDA represents the dimensionless diffusion number (mesh Fourier number):
  //   lambda = (kappa * dt) / (dx^2)
  // where:
  //   - kappa: thermal diffusivity
  //   - dt   : physical time-step size
  //   - dx   : spatial grid spacing
  // In explicit methods, numerical stability requires lambda <= 0.25 in 2D.
  // In implicit Backward Euler, the method is unconditionally stable for any lambda > 0.
  double LAMBDA = 0.05;

  // Jacobi iteration parameters for the implicit linear solve
  int MAX_JACOBI_ITERS = 25;   // Maximum inner solver iterations per time step
  double TOL = 1e-5;           // Convergence tolerance for Jacobi solver

  // Optional command-line arguments:
  // ./heat_serial_implicit [N] [STEPS] [LAMBDA] [MAX_JACOBI_ITERS] [TOL]
  if (argc >= 2) N = atoi(argv[1]);
  if (argc >= 3) STEPS = atoi(argv[2]);
  if (argc >= 4) LAMBDA = atof(argv[3]);
  if (argc >= 5) MAX_JACOBI_ITERS = atoi(argv[4]);
  if (argc >= 6) TOL = atof(argv[5]);

  if (N < 3 || STEPS < 0 || LAMBDA <= 0.0) {
    cerr << "Error: Invalid arguments. N >= 3, STEPS >= 0, LAMBDA > 0.\n";
    return 1;
  }

  /*
   * Implicit Time Stepping (Backward Euler):
   * ----------------------------------------------------
   * Continuous: du/dt = kappa * laplacian(u)
   *
   * Discrete (Backward Euler at time step n+1 with lambda = kappa*dt / dx^2):
   * u^{n+1}_{i,j} = u^n_{i,j} + lambda * (u^{n+1}_{i-1,j} + u^{n+1}_{i+1,j} +
   *                                       u^{n+1}_{i,j-1} + u^{n+1}_{i,j+1} -
   *                                       4 * u^{n+1}_{i,j})
   *
   * Rearranging into linear system A * u^{n+1} = u^n:
   * (1 + 4*lambda) * u^{n+1}_{i,j} - lambda * (u^{n+1}_{i-1,j} + u^{n+1}_{i+1,j} +
   *                                          u^{n+1}_{i,j-1} + u^{n+1}_{i,j+1}) = u^n_{i,j}
   *
   * The system matrix A has diagonal D = (1 + 4*lambda)
   * and off-diagonals -lambda for adjacent orthogonal cells.
   *
   * Jacobi Iteration formula to solve this system at each time step:
   * u^{(k+1)}_{i,j} = [ u^n_{i,j} + lambda * (u^{(k)}_{i-1,j} + u^{(k)}_{i+1,j} +
   *                                           u^{(k)}_{i,j-1} + u^{(k)}_{i,j+1}) ] / (1 + 4*lambda)
   */

  const double diag = 1.0 + 4.0 * LAMBDA;
  const double invDiag = 1.0 / diag;

  // Flat 1D vectors for cache efficiency
  // u_prev: temperature at time step n (RHS of the linear system)
  // u_curr: current estimate u^(k) in Jacobi iteration
  // u_next: updated estimate u^(k+1) in Jacobi iteration
  vector<double> u_prev(N * N, 0.0);
  vector<double> u_curr(N * N, 0.0);
  vector<double> u_next(N * N, 0.0);

  // Initial condition: hot cell at the center
  u_prev[idx(N / 2, N / 2, N)] = 100.0;
  u_curr[idx(N / 2, N / 2, N)] = 100.0;
  u_next[idx(N / 2, N / 2, N)] = 100.0;

  cout << "========================================================\n";
  cout << "   2D Heat Equation - Implicit Time Stepping (Serial)   \n";
  cout << "   Method: Backward Euler with Jacobi Iterative Solver   \n";
  cout << "========================================================\n";
  cout << "Grid size (N x N)    : " << N << " x " << N << "\n";
  cout << "Time steps           : " << STEPS << "\n";
  cout << "Lambda (diffusion)   : " << LAMBDA << "  [lambda = kappa*dt/dx^2]\n";
  cout << "Max Jacobi Iters/Step: " << MAX_JACOBI_ITERS << "\n";
  cout << "Convergence Tol      : " << TOL << "\n";
  cout << "Running simulation...\n";

  auto start = chrono::steady_clock::now();

  long long totalJacobiIters = 0;
  int nonConvergedSteps = 0;
  double maxResidualOverall = 0.0;

  for (int t = 0; t < STEPS; t++) {
    // Initial guess for the implicit solution u^{n+1} is the solution from time step n
    // (u_curr is already initialized to u_prev from the previous step)

    bool converged = false;
    double lastMaxDiff = 0.0;
    int iter = 0;

    for (; iter < MAX_JACOBI_ITERS; iter++) {
      double maxDiff = 0.0;

      // Update interior cells using Jacobi relaxation
      for (int i = 1; i < N - 1; i++) {
        const size_t rowOffset = i * N;
        const size_t upOffset = (i - 1) * N;
        const size_t downOffset = (i + 1) * N;

        for (int j = 1; j < N - 1; j++) {
          double neighborsSum = u_curr[upOffset + j] +
                                u_curr[downOffset + j] +
                                u_curr[rowOffset + j - 1] +
                                u_curr[rowOffset + j + 1];

          double newVal = (u_prev[rowOffset + j] + LAMBDA * neighborsSum) * invDiag;
          u_next[rowOffset + j] = newVal;

          double diff = std::abs(newVal - u_curr[rowOffset + j]);
          if (diff > maxDiff) {
            maxDiff = diff;
          }
        }
      }

      // Prepare for next Jacobi iteration
      u_curr.swap(u_next);
      lastMaxDiff = maxDiff;

      // Check convergence of inner solver
      if (maxDiff < TOL) {
        converged = true;
        iter++;
        break;
      }
    }

    if (!converged) {
      nonConvergedSteps++;
      if (nonConvergedSteps <= 5) {
        cerr << "Warning: Jacobi did not converge at time step " << t
             << " (reached max " << MAX_JACOBI_ITERS
             << " iters, last diff: " << lastMaxDiff << " >= TOL: " << TOL << ")\n";
      } else if (nonConvergedSteps == 6) {
        cerr << "Warning: Further convergence warnings suppressed...\n";
      }
    }

    totalJacobiIters += iter;

    // Calculate maximum absolute residual r = b - A*u^{n+1} on the FINAL time step only
    if (t == STEPS - 1) {
      double stepMaxResidual = 0.0;
      for (int i = 1; i < N - 1; i++) {
        const size_t rowOffset = i * N;
        const size_t upOffset = (i - 1) * N;
        const size_t downOffset = (i + 1) * N;

        for (int j = 1; j < N - 1; j++) {
          double Au = diag * u_curr[rowOffset + j] -
                      LAMBDA * (u_curr[upOffset + j] + u_curr[downOffset + j] +
                                u_curr[rowOffset + j - 1] + u_curr[rowOffset + j + 1]);
          double res = std::abs(u_prev[rowOffset + j] - Au);
          if (res > stepMaxResidual) {
            stepMaxResidual = res;
          }
        }
      }
      maxResidualOverall = stepMaxResidual;
    }

    // Time step completed: advance u_prev to the newly converged solution u^{n+1}
    u_prev = u_curr;
  }

  auto end = chrono::steady_clock::now();
  double elapsed = chrono::duration<double>(end - start).count();

  // Compute final grid validation metrics (heat conservation & bounds)
  double maxTemp = u_prev[0];
  double minTemp = u_prev[0];
  double totalHeat = 0.0;
  for (size_t k = 0; k < (size_t)N * N; k++) {
    double val = u_prev[k];
    if (val > maxTemp) maxTemp = val;
    if (val < minTemp) minTemp = val;
    totalHeat += val;
  }

  cout << fixed << setprecision(6);
  cout << "========================================================\n";
  cout << "Execution time       : " << elapsed << " seconds\n";
  if (STEPS > 0) {
    cout << "Time per time step   : " << (elapsed / STEPS) * 1000.0 << " ms\n";
    cout << "Avg Jacobi iters/step: " << (double)totalJacobiIters / STEPS << "\n";
  }
  cout << "Total Jacobi iters   : " << totalJacobiIters << "\n";
  cout << "Non-converged steps  : " << nonConvergedSteps << " / " << STEPS;
  if (nonConvergedSteps == 0) {
    cout << " (100% converged)\n";
  } else {
    cout << " (WARNING: " << fixed << setprecision(2)
         << ((double)nonConvergedSteps / STEPS * 100.0) << "% failed to reach TOL)\n";
    cout << fixed << setprecision(6);
  }
  cout << "Max absolute residual: " << scientific << maxResidualOverall << defaultfloat << "\n";
  cout << fixed << setprecision(6);
  cout << "--------------------------------------------------------\n";
  cout << "Validation Metrics (Physics & Stability Check):\n";
  cout << "  - Center Temperature  : " << u_prev[idx(N / 2, N / 2, N)] << "\n";
  cout << "  - Maximum Temperature : " << maxTemp << "\n";
  cout << "  - Minimum Temperature : " << minTemp;
  if (minTemp < -1e-12) {
    cout << " (WARNING: Negative temperature detected!)\n";
  } else {
    cout << " (Stable, >= 0)\n";
  }
  cout << "  - Total Heat Energy   : " << totalHeat
       << " (Initial: 100.000000, Loss: " << (100.0 - totalHeat) << ")\n";
  cout << "--------------------------------------------------------\n";

  // Automatically save baseline output files for verification
#if defined(_WIN32)
  _mkdir("results");
#else
  mkdir("results", 0777);
#endif

  string baselineBin = "results/baseline_implicit_N" + to_string(N) + "_S" + to_string(STEPS) + ".bin";
  string baselineTxt = "results/baseline_implicit_metrics_N" + to_string(N) + "_S" + to_string(STEPS) + ".txt";

  ofstream binFile(baselineBin, ios::binary);
  if (binFile.is_open()) {
    binFile.write(reinterpret_cast<const char*>(u_prev.data()), sizeof(double) * N * N);
    binFile.close();
    cout << "Saved baseline grid   : " << baselineBin << " ("
         << fixed << setprecision(2) << (sizeof(double) * N * N) / (1024.0 * 1024.0) << " MB)\n";
  } else {
    cerr << "Warning: Could not open " << baselineBin << " to save baseline grid.\n";
  }

  ofstream txtFile(baselineTxt);
  if (txtFile.is_open()) {
    txtFile << fixed << setprecision(10);
    txtFile << "N: " << N << "\n";
    txtFile << "STEPS: " << STEPS << "\n";
    txtFile << "LAMBDA: " << LAMBDA << "\n";
    txtFile << "MAX_JACOBI_ITERS: " << MAX_JACOBI_ITERS << "\n";
    txtFile << "TOL: " << scientific << TOL << fixed << "\n";
    txtFile << "Center_Temperature: " << u_prev[idx(N / 2, N / 2, N)] << "\n";
    txtFile << "Max_Temperature: " << maxTemp << "\n";
    txtFile << "Min_Temperature: " << minTemp << "\n";
    txtFile << "Total_Heat_Energy: " << totalHeat << "\n";
    txtFile << "Total_Jacobi_Iters: " << totalJacobiIters << "\n";
    txtFile << "Non_Converged_Steps: " << nonConvergedSteps << "\n";
    txtFile << "Max_Absolute_Residual: " << scientific << maxResidualOverall << fixed << "\n";
    txtFile << "Execution_Time_Seconds: " << elapsed << "\n";
    txtFile.close();
    cout << "Saved baseline metrics: " << baselineTxt << "\n";
  }

  cout << "========================================================\n";

  return 0;
}
