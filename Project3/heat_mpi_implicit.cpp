#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

using namespace std;

// Convert local row (with halos) and global col to 1D flat index
inline size_t idx(size_t i, size_t j, size_t n) { return i * n + j; }

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Default parameters matching heat_serial_implicit.cpp
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
  // In implicit Backward Euler, the method is unconditionally stable for any
  // lambda > 0.
  double LAMBDA = 0.05;

  // Jacobi iteration parameters for the implicit linear solve
  int MAX_JACOBI_ITERS = 25; // Maximum inner solver iterations per time step
  double TOL = 1e-5;         // Convergence tolerance for Jacobi solver

  // Optional command-line arguments:
  // mpirun -np <p> ./heat_mpi_implicit [N] [STEPS] [LAMBDA] [MAX_JACOBI_ITERS]
  // [TOL]
  if (argc >= 2)
    N = atoi(argv[1]);
  if (argc >= 3)
    STEPS = atoi(argv[2]);
  if (argc >= 4)
    LAMBDA = atof(argv[3]);
  if (argc >= 5)
    MAX_JACOBI_ITERS = atoi(argv[4]);
  if (argc >= 6)
    TOL = atof(argv[5]);

  if (N < 3 || STEPS < 0 || LAMBDA <= 0.0 || size > N) {
    if (rank == 0) {
      cerr << "Error: Invalid arguments. N >= 3, STEPS >= 0, LAMBDA > 0, "
              "processes <= N.\n";
    }
    MPI_Finalize();
    return 1;
  }

  // Divide rows among MPI processes (1D row decomposition)
  int base = N / size;
  int remainder = N % size;

  int localRows = base + (rank < remainder ? 1 : 0);
  int startRow = rank * base + min(rank, remainder);

  const double diag = 1.0 + 4.0 * LAMBDA;
  const double invDiag = 1.0 / diag;

  // Each rank allocates (localRows + 2) rows to include top and bottom halo
  // rows u_prev: temperature at time step n (RHS of the linear system) u_curr:
  // current estimate u^(k) in Jacobi iteration (exchanged via halos) u_next:
  // updated estimate u^(k+1) in Jacobi iteration
  vector<double> u_prev((localRows + 2) * N, 0.0);
  vector<double> u_curr((localRows + 2) * N, 0.0);
  vector<double> u_next((localRows + 2) * N, 0.0);

  // Initial condition: hot cell at the center
  int center = N / 2;
  if (center >= startRow && center < startRow + localRows) {
    int localCenter = center - startRow + 1;
    u_prev[idx(localCenter, center, N)] = 100.0;
    u_curr[idx(localCenter, center, N)] = 100.0;
    u_next[idx(localCenter, center, N)] = 100.0;
  }

  int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
  int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

  if (rank == 0) {
    cout << "========================================================\n";
    cout << "     2D Heat Equation - Implicit Time Stepping (MPI)    \n";
    cout << "     Method: Backward Euler with Jacobi Iterative Solver\n";
    cout << "========================================================\n";
    cout << "Grid size (N x N)    : " << N << " x " << N << "\n";
    cout << "Time steps           : " << STEPS << "\n";
    cout << "Lambda (diffusion)   : " << LAMBDA
         << "  [lambda = kappa*dt/dx^2]\n";
    cout << "Max Jacobi Iters/Step: " << MAX_JACOBI_ITERS << "\n";
    cout << "Convergence Tol      : " << TOL << "\n";
    cout << "MPI processes        : " << size << "\n";
    cout << "Running simulation...\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  long long totalJacobiIters = 0;
  int nonConvergedSteps = 0;
  double localMaxResidualOverall = 0.0;

  for (int t = 0; t < STEPS; t++) {
    // Initial guess for u^{n+1} is the solution from time step n (u_curr)

    bool converged = false;
    double lastGlobalMaxDiff = 0.0;
    int iter = 0;

    for (; iter < MAX_JACOBI_ITERS; iter++) {

      // Halo exchange for u_curr:
      // 1. Send first owned row upward; receive lower neighbor into bottom halo
      MPI_Sendrecv(&u_curr[idx(1, 0, N)], N, MPI_DOUBLE, up, 0,
                   &u_curr[idx(localRows + 1, 0, N)], N, MPI_DOUBLE, down, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // 2. Send last owned row downward; receive upper neighbor into top halo
      MPI_Sendrecv(&u_curr[idx(localRows, 0, N)], N, MPI_DOUBLE, down, 1,
                   &u_curr[idx(0, 0, N)], N, MPI_DOUBLE, up, 1, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);

      double localMaxDiff = 0.0;

      // Update owned interior cells using Jacobi relaxation
      for (int i = 1; i <= localRows; i++) {
        int globalRow = startRow + i - 1;

        // Keep physical top and bottom boundaries fixed at 0.0
        if (globalRow == 0 || globalRow == N - 1)
          continue;

        const size_t rowOffset = i * N;
        const size_t upOffset = (i - 1) * N;
        const size_t downOffset = (i + 1) * N;

        for (int j = 1; j < N - 1; j++) {
          double neighborsSum = u_curr[upOffset + j] + u_curr[downOffset + j] +
                                u_curr[rowOffset + j - 1] +
                                u_curr[rowOffset + j + 1];

          double newVal =
              (u_prev[rowOffset + j] + LAMBDA * neighborsSum) * invDiag;
          u_next[rowOffset + j] = newVal;

          double diff = std::abs(newVal - u_curr[rowOffset + j]);
          if (diff > localMaxDiff) {
            localMaxDiff = diff;
          }
        }
      }

      // Prepare for next Jacobi iteration
      u_curr.swap(u_next);

      // Check global convergence across all MPI processes
      double globalMaxDiff = 0.0;
      MPI_Allreduce(&localMaxDiff, &globalMaxDiff, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
      lastGlobalMaxDiff = globalMaxDiff;

      if (globalMaxDiff < TOL) {
        converged = true;
        iter++;
        break;
      }
    }

    if (!converged) {
      nonConvergedSteps++;
      if (rank == 0) {
        if (nonConvergedSteps <= 5) {
          cerr << "Warning: Jacobi did not converge at time step " << t
               << " (reached max " << MAX_JACOBI_ITERS
               << " iters, last diff: " << lastGlobalMaxDiff
               << " >= TOL: " << TOL << ")\n";
        } else if (nonConvergedSteps == 6) {
          cerr << "Warning: Further convergence warnings suppressed...\n";
        }
      }
    }

    totalJacobiIters += iter;

    // Calculate maximum absolute residual r = b - A*u^{n+1} on the FINAL time
    // step only
    if (t == STEPS - 1) {
      // Halo exchange for the finalized u^{n+1} (u_curr) to compute accurate
      // residuals
      MPI_Sendrecv(&u_curr[idx(1, 0, N)], N, MPI_DOUBLE, up, 0,
                   &u_curr[idx(localRows + 1, 0, N)], N, MPI_DOUBLE, down, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Sendrecv(&u_curr[idx(localRows, 0, N)], N, MPI_DOUBLE, down, 1,
                   &u_curr[idx(0, 0, N)], N, MPI_DOUBLE, up, 1, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);

      double stepLocalMaxResidual = 0.0;
      for (int i = 1; i <= localRows; i++) {
        int globalRow = startRow + i - 1;
        if (globalRow == 0 || globalRow == N - 1)
          continue;

        const size_t rowOffset = i * N;
        const size_t upOffset = (i - 1) * N;
        const size_t downOffset = (i + 1) * N;

        for (int j = 1; j < N - 1; j++) {
          double Au =
              diag * u_curr[rowOffset + j] -
              LAMBDA * (u_curr[upOffset + j] + u_curr[downOffset + j] +
                        u_curr[rowOffset + j - 1] + u_curr[rowOffset + j + 1]);
          double res = std::abs(u_prev[rowOffset + j] - Au);
          if (res > stepLocalMaxResidual) {
            stepLocalMaxResidual = res;
          }
        }
      }
      localMaxResidualOverall = stepLocalMaxResidual;
    }

    // Time step completed: advance u_prev to the newly converged solution
    // u^{n+1}
    u_prev = u_curr;
  }

  double localTime = MPI_Wtime() - start;
  double maxTime = 0.0;
  MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // Global reduction of maximum absolute residual
  double globalMaxResidual = 0.0;
  MPI_Reduce(&localMaxResidualOverall, &globalMaxResidual, 1, MPI_DOUBLE,
             MPI_MAX, 0, MPI_COMM_WORLD);

  // Compute final grid validation metrics across owned cells
  double localMaxTemp = u_prev[idx(1, 0, N)];
  double localMinTemp = u_prev[idx(1, 0, N)];
  double localTotalHeat = 0.0;
  for (int i = 1; i <= localRows; i++) {
    for (int j = 0; j < N; j++) {
      double val = u_prev[idx(i, j, N)];
      if (val > localMaxTemp)
        localMaxTemp = val;
      if (val < localMinTemp)
        localMinTemp = val;
      localTotalHeat += val;
    }
  }

  double globalMaxTemp = 0.0, globalMinTemp = 0.0, globalTotalHeat = 0.0;
  MPI_Reduce(&localMaxTemp, &globalMaxTemp, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&localMinTemp, &globalMinTemp, 1, MPI_DOUBLE, MPI_MIN, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&localTotalHeat, &globalTotalHeat, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  // Find center temperature for reporting
  double centerTemp = 0.0;
  if (center >= startRow && center < startRow + localRows) {
    centerTemp = u_prev[idx(center - startRow + 1, center, N)];
  }
  double globalCenterTemp = 0.0;
  MPI_Reduce(&centerTemp, &globalCenterTemp, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    cout << fixed << setprecision(6);
    cout << "========================================================\n";
    cout << "Execution time       : " << maxTime << " seconds\n";
    if (STEPS > 0) {
      cout << "Time per time step   : " << (maxTime / STEPS) * 1000.0
           << " ms\n";
      cout << "Avg Jacobi iters/step: " << (double)totalJacobiIters / STEPS
           << "\n";
    }
    cout << "Total Jacobi iters   : " << totalJacobiIters << "\n";
    cout << "Non-converged steps  : " << nonConvergedSteps << " / " << STEPS;
    if (nonConvergedSteps == 0) {
      cout << " (100% converged)\n";
    } else {
      cout << " (WARNING: " << fixed << setprecision(2)
           << ((double)nonConvergedSteps / STEPS * 100.0)
           << "% failed to reach TOL)\n";
      cout << fixed << setprecision(6);
    }
    cout << "Max absolute residual: " << scientific << globalMaxResidual
         << defaultfloat << "\n";
    cout << fixed << setprecision(6);
    cout << "--------------------------------------------------------\n";
    cout << "Validation Metrics (Physics & Stability Check):\n";
    cout << "  - Center Temperature  : " << globalCenterTemp << "\n";
    cout << "  - Maximum Temperature : " << globalMaxTemp << "\n";
    cout << "  - Minimum Temperature : " << globalMinTemp;
    if (globalMinTemp < -1e-12) {
      cout << " (WARNING: Negative temperature detected!)\n";
    } else {
      cout << " (Stable, >= 0)\n";
    }
    cout << "  - Total Heat Energy   : " << globalTotalHeat
         << " (Initial: 100.000000, Loss: " << (100.0 - globalTotalHeat)
         << ")\n";
    cout << "MPI processes        : " << size << "\n";
    cout << "========================================================\n";
  }

  // Automatic verification against serial baseline if baseline file exists
  int baselineExists = 0;
  string baselineBin = "results/baseline_implicit_N" + to_string(N) + "_S" + to_string(STEPS) + ".bin";
  if (rank == 0) {
    ifstream testFile(baselineBin, ios::binary);
    if (testFile.good()) {
      baselineExists = 1;
    }
  }
  MPI_Bcast(&baselineExists, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (baselineExists) {
    vector<double> localOwned(localRows * N);
    for (int i = 1; i <= localRows; i++) {
      for (int j = 0; j < N; j++) {
        localOwned[(i - 1) * N + j] = u_prev[idx(i, j, N)];
      }
    }

    vector<int> recvcounts(size), displs(size);
    for (int r = 0; r < size; r++) {
      int r_rows = base + (r < remainder ? 1 : 0);
      recvcounts[r] = r_rows * N;
      displs[r] = (r * base + min(r, remainder)) * N;
    }

    vector<double> fullGrid;
    if (rank == 0) fullGrid.resize((size_t)N * N);

    MPI_Gatherv(localOwned.data(), localRows * N, MPI_DOUBLE,
                fullGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      vector<double> baselineGrid((size_t)N * N);
      ifstream binIn(baselineBin, ios::binary);
      binIn.read(reinterpret_cast<char*>(baselineGrid.data()), sizeof(double) * N * N);
      binIn.close();

      double maxDiff = 0.0;
      for (size_t k = 0; k < (size_t)N * N; k++) {
        double d = std::abs(fullGrid[k] - baselineGrid[k]);
        if (d > maxDiff) maxDiff = d;
      }

      cout << "Baseline Solution Verification:\n";
      cout << "  - Baseline File       : " << baselineBin << "\n";
      cout << "  - Max Absolute Diff   : " << scientific << maxDiff << defaultfloat << "\n";
      if (maxDiff < 1e-9) {
        cout << "  - Verification Result : PASSED (Bit-for-bit identical to Serial Baseline)\n";
      } else {
        cout << "  - Verification Result : WARNING (Discrepancy detected!)\n";
      }
      cout << "========================================================\n";
    }
  }

  MPI_Finalize();
  return 0;
}
