#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

// Convert 2D grid coordinates (i, j) to flat 1D index
inline size_t idx(size_t i, size_t j, size_t n) {
  return i * n + j;
}

int main(int argc, char* argv[]) {
  // Default parameters matching heat_serial.cpp
  int N = 3000;
  int STEPS = 1000;
  double ALPHA = 0.05;

  // Jacobi iteration parameters for the implicit linear solve
  int MAX_JACOBI_ITERS = 25;   // Maximum inner solver iterations per time step
  double TOL = 1e-5;           // Convergence tolerance for Jacobi solver

  // Optional command-line arguments:
  // ./heat_serial_implicit [N] [STEPS] [ALPHA] [MAX_JACOBI_ITERS] [TOL]
  if (argc >= 2) N = atoi(argv[1]);
  if (argc >= 3) STEPS = atoi(argv[2]);
  if (argc >= 4) ALPHA = atof(argv[3]);
  if (argc >= 5) MAX_JACOBI_ITERS = atoi(argv[4]);
  if (argc >= 6) TOL = atof(argv[5]);

  if (N < 3 || STEPS < 0 || ALPHA <= 0.0) {
    cerr << "Error: Invalid arguments. N >= 3, STEPS >= 0, ALPHA > 0.\n";
    return 1;
  }

  /*
   * Implicit Time Stepping (Backward Euler):
   * ----------------------------------------------------
   * Continuous: du/dt = alpha * laplacian(u)
   *
   * Discrete (Backward Euler at time n+1):
   * u^{n+1}_{i,j} = u^n_{i,j} + alpha * (u^{n+1}_{i-1,j} + u^{n+1}_{i+1,j} +
   *                                      u^{n+1}_{i,j-1} + u^{n+1}_{i,j+1} -
   *                                      4 * u^{n+1}_{i,j})
   *
   * Rearranging into linear system A * u^{n+1} = u^n:
   * (1 + 4*alpha) * u^{n+1}_{i,j} - alpha * (u^{n+1}_{i-1,j} + u^{n+1}_{i+1,j} +
   *                                         u^{n+1}_{i,j-1} + u^{n+1}_{i,j+1}) = u^n_{i,j}
   *
   * The Jacobian matrix J = dF/du^{n+1} has diagonal D = (1 + 4*alpha)
   * and off-diagonals -alpha for adjacent cells.
   *
   * Jacobi Iteration formula to solve this system at each time step:
   * u^{(k+1)}_{i,j} = [ u^n_{i,j} + alpha * (u^{(k)}_{i-1,j} + u^{(k)}_{i+1,j} +
   *                                          u^{(k)}_{i,j-1} + u^{(k)}_{i,j+1}) ] / (1 + 4*alpha)
   */

  const double diag = 1.0 + 4.0 * ALPHA;
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
  cout << "Alpha (diffusion)    : " << ALPHA << "\n";
  cout << "Max Jacobi Iters/Step: " << MAX_JACOBI_ITERS << "\n";
  cout << "Convergence Tol      : " << TOL << "\n";
  cout << "Running simulation...\n";

  auto start = chrono::steady_clock::now();

  long long totalJacobiIters = 0;

  for (int t = 0; t < STEPS; t++) {
    // Initial guess for the implicit solution u^{n+1} is the solution from time step n
    // (u_curr is already initialized to u_prev from the previous step)

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

          double newVal = (u_prev[rowOffset + j] + ALPHA * neighborsSum) * invDiag;
          u_next[rowOffset + j] = newVal;

          double diff = std::abs(newVal - u_curr[rowOffset + j]);
          if (diff > maxDiff) {
            maxDiff = diff;
          }
        }
      }

      // Prepare for next Jacobi iteration
      u_curr.swap(u_next);

      // Check convergence of inner solver
      if (maxDiff < TOL) {
        iter++;
        break;
      }
    }

    totalJacobiIters += iter;

    // Time step completed: advance u_prev to the newly converged solution u^{n+1}
    u_prev = u_curr;
  }

  auto end = chrono::steady_clock::now();
  double elapsed = chrono::duration<double>(end - start).count();

  cout << fixed << setprecision(6);
  cout << "========================================================\n";
  cout << "Execution time       : " << elapsed << " seconds\n";
  if (STEPS > 0) {
    cout << "Time per time step   : " << (elapsed / STEPS) * 1000.0 << " ms\n";
    cout << "Avg Jacobi iters/step: " << (double)totalJacobiIters / STEPS << "\n";
  }
  cout << "Total Jacobi iters   : " << totalJacobiIters << "\n";
  cout << "Center temperature   : " << u_prev[idx(N / 2, N / 2, N)] << "\n";
  cout << "========================================================\n";

  return 0;
}
