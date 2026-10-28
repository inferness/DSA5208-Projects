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
  // mpirun -np <p> ./heat_mpi_block_nonblocking_implicit [N] [STEPS] [LAMBDA]
  // [MAX_JACOBI_ITERS] [TOL]
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

  if (N < 3 || STEPS < 0 || LAMBDA <= 0.0) {
    if (rank == 0) {
      cerr << "Error: Invalid arguments. N >= 3, STEPS >= 0, LAMBDA > 0.\n";
    }
    MPI_Finalize();
    return 1;
  }

  // Create 2D Cartesian process grid
  int dims[2] = {0, 0};
  MPI_Dims_create(size, 2, dims);

  int periods[2] = {0, 0}; // Non-periodic domain (fixed Dirichlet boundaries)
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

  int coords[2];
  MPI_Cart_coords(cart_comm, rank, 2, coords);

  // Determine neighboring ranks (returns MPI_PROC_NULL for physical boundary edges)
  int up, down, left, right;
  MPI_Cart_shift(cart_comm, 0, 1, &up, &down);
  MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

  // Divide rows along dimension 0
  int base_r = N / dims[0];
  int rem_r = N % dims[0];
  int localRows = base_r + (coords[0] < rem_r ? 1 : 0);
  int startRow = coords[0] * base_r + min(coords[0], rem_r);

  // Divide columns along dimension 1
  int base_c = N / dims[1];
  int rem_c = N % dims[1];
  int localCols = base_c + (coords[1] < rem_c ? 1 : 0);
  int startCol = coords[1] * base_c + min(coords[1], rem_c);

  const double diag = 1.0 + 4.0 * LAMBDA;
  const double invDiag = 1.0 / diag;

  // Local grid dimensions with halo layer (1 cell on each of the 4 borders)
  // Leading dimension LD = localCols + 2
  int LD = localCols + 2;
  vector<double> u_prev((localRows + 2) * LD, 0.0);
  vector<double> u_curr((localRows + 2) * LD, 0.0);
  vector<double> u_next((localRows + 2) * LD, 0.0);

  // Convert 2D local index to 1D index
  auto idx = [LD](int i, int j) { return (size_t)i * LD + j; };

  // Initialize the hot center if this rank owns it
  int center = N / 2;
  if (center >= startRow && center < startRow + localRows &&
      center >= startCol && center < startCol + localCols) {
    int localCenterRow = center - startRow + 1;
    int localCenterCol = center - startCol + 1;
    u_prev[idx(localCenterRow, localCenterCol)] = 100.0;
    u_curr[idx(localCenterRow, localCenterCol)] = 100.0;
    u_next[idx(localCenterRow, localCenterCol)] = 100.0;
  }

  // Define derived MPI datatype for sending/receiving non-contiguous vertical column halos
  MPI_Datatype col_type;
  MPI_Type_vector(localRows, 1, LD, MPI_DOUBLE, &col_type);
  MPI_Type_commit(&col_type);

  // Precompute interior loop bounds once outside the time-step loop (keeping global boundaries at 0.0)
  int i_start = (startRow == 0) ? 2 : 1;
  int i_end = (startRow + localRows == N) ? localRows - 1 : localRows;

  int j_start = (startCol == 0) ? 2 : 1;
  int j_end = (startCol + localCols == N) ? localCols - 1 : localCols;

  // Precompute overlap interior bounds (strictly independent of halo rows 0, localRows+1 and cols 0, localCols+1)
  int i_overlap_start = max(i_start, 2);
  int i_overlap_end = min(i_end, localRows - 1);

  int j_overlap_start = max(j_start, 2);
  int j_overlap_end = min(j_end, localCols - 1);

  if (rank == 0) {
    cout << "========================================================\n";
    cout << "  2D Heat Equation - Implicit 2D Block (Non-blocking)  \n";
    cout << "  Method: Backward Euler with Jacobi Iterative Solver   \n";
    cout << "  Overlap: Interior Block Relaxed during Halo Transfer  \n";
    cout << "========================================================\n";
    cout << "Grid size (N x N)    : " << N << " x " << N << "\n";
    cout << "Time steps           : " << STEPS << "\n";
    cout << "Lambda (diffusion)   : " << LAMBDA
         << "  [lambda = kappa*dt/dx^2]\n";
    cout << "Max Jacobi Iters/Step: " << MAX_JACOBI_ITERS << "\n";
    cout << "Convergence Tol      : " << TOL << "\n";
    cout << "MPI processes        : " << size
         << " (2D Process Grid: " << dims[0] << " x " << dims[1] << ")\n";
    cout << "Running simulation...\n";
  }

  MPI_Barrier(cart_comm);
  double start = MPI_Wtime();

  long long totalJacobiIters = 0;
  int nonConvergedSteps = 0;
  double localMaxResidualOverall = 0.0;
  double local_comp_time = 0.0;
  double local_comm_time = 0.0;

  for (int t = 0; t < STEPS; t++) {
    bool converged = false;
    double lastGlobalMaxDiff = 0.0;
    int iter = 0;

    for (; iter < MAX_JACOBI_ITERS; iter++) {

      double t_comm0 = MPI_Wtime();
      // 1. Post 8 non-blocking halo transfers
      MPI_Request reqs[8];

      // North / South (contiguous rows)
      MPI_Irecv(&u_curr[idx(0, 1)], localCols, MPI_DOUBLE, up, 1, cart_comm, &reqs[0]);
      MPI_Irecv(&u_curr[idx(localRows + 1, 1)], localCols, MPI_DOUBLE, down, 0, cart_comm, &reqs[1]);
      MPI_Isend(&u_curr[idx(1, 1)], localCols, MPI_DOUBLE, up, 0, cart_comm, &reqs[2]);
      MPI_Isend(&u_curr[idx(localRows, 1)], localCols, MPI_DOUBLE, down, 1, cart_comm, &reqs[3]);

      // West / East (non-contiguous strided columns)
      MPI_Irecv(&u_curr[idx(1, 0)], 1, col_type, left, 3, cart_comm, &reqs[4]);
      MPI_Irecv(&u_curr[idx(1, localCols + 1)], 1, col_type, right, 2, cart_comm, &reqs[5]);
      MPI_Isend(&u_curr[idx(1, 1)], 1, col_type, left, 2, cart_comm, &reqs[6]);
      MPI_Isend(&u_curr[idx(1, localCols)], 1, col_type, right, 3, cart_comm, &reqs[7]);
      local_comm_time += (MPI_Wtime() - t_comm0);

      double t_comp0 = MPI_Wtime();
      double localMaxDiff = 0.0;

      // 2. Computation-Communication Overlap:
      // Update interior cells [i_overlap_start..i_overlap_end] x [j_overlap_start..j_overlap_end]
      // These cells strictly depend on owned data, NOT on ghost halos!
      if (i_overlap_start <= i_overlap_end && j_overlap_start <= j_overlap_end) {
        for (int i = i_overlap_start; i <= i_overlap_end; i++) {
          const size_t rowOffset = (size_t)i * LD;
          const size_t upOffset = (size_t)(i - 1) * LD;
          const size_t downOffset = (size_t)(i + 1) * LD;

          for (int j = j_overlap_start; j <= j_overlap_end; j++) {
            double neighborsSum = u_curr[upOffset + j] + u_curr[downOffset + j] +
                                  u_curr[rowOffset + j - 1] +
                                  u_curr[rowOffset + j + 1];

            double newVal = (u_prev[rowOffset + j] + LAMBDA * neighborsSum) * invDiag;
            u_next[rowOffset + j] = newVal;

            double diff = std::abs(newVal - u_curr[rowOffset + j]);
            if (diff > localMaxDiff) {
              localMaxDiff = diff;
            }
          }
        }
      }
      local_comp_time += (MPI_Wtime() - t_comp0);

      // 3. Wait for all 8 halo communications to complete
      double t_wait = MPI_Wtime();
      MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
      local_comm_time += (MPI_Wtime() - t_wait);

      double t_comp1 = MPI_Wtime();
      // 4. Update the boundary cells of the local block using newly arrived halos:
      // 4a. Top owned row (i = 1) across all columns j in [j_start, j_end]
      if (i_start <= 1 && 1 <= i_end) {
        const size_t rowOffset = 1 * LD;
        const size_t upOffset = 0 * LD;
        const size_t downOffset = 2 * LD;
        for (int j = j_start; j <= j_end; j++) {
          double neighborsSum = u_curr[upOffset + j] + u_curr[downOffset + j] +
                                u_curr[rowOffset + j - 1] +
                                u_curr[rowOffset + j + 1];

          double newVal = (u_prev[rowOffset + j] + LAMBDA * neighborsSum) * invDiag;
          u_next[rowOffset + j] = newVal;

          double diff = std::abs(newVal - u_curr[rowOffset + j]);
          if (diff > localMaxDiff) localMaxDiff = diff;
        }
      }

      // 4b. Bottom owned row (i = localRows) across all columns j in [j_start, j_end] (if localRows > 1)
      if (localRows > 1 && i_start <= localRows && localRows <= i_end) {
        const size_t rowOffset = (size_t)localRows * LD;
        const size_t upOffset = (size_t)(localRows - 1) * LD;
        const size_t downOffset = (size_t)(localRows + 1) * LD;
        for (int j = j_start; j <= j_end; j++) {
          double neighborsSum = u_curr[upOffset + j] + u_curr[downOffset + j] +
                                u_curr[rowOffset + j - 1] +
                                u_curr[rowOffset + j + 1];

          double newVal = (u_prev[rowOffset + j] + LAMBDA * neighborsSum) * invDiag;
          u_next[rowOffset + j] = newVal;

          double diff = std::abs(newVal - u_curr[rowOffset + j]);
          if (diff > localMaxDiff) localMaxDiff = diff;
        }
      }

      // 4c. Left owned column (j = 1) for interior rows i in [i_overlap_start, i_overlap_end]
      if (j_start <= 1 && 1 <= j_end && i_overlap_start <= i_overlap_end) {
        for (int i = i_overlap_start; i <= i_overlap_end; i++) {
          const size_t rowOffset = (size_t)i * LD;
          const size_t upOffset = (size_t)(i - 1) * LD;
          const size_t downOffset = (size_t)(i + 1) * LD;
          int j = 1;

          double neighborsSum = u_curr[upOffset + j] + u_curr[downOffset + j] +
                                u_curr[rowOffset + j - 1] +
                                u_curr[rowOffset + j + 1];

          double newVal = (u_prev[rowOffset + j] + LAMBDA * neighborsSum) * invDiag;
          u_next[rowOffset + j] = newVal;

          double diff = std::abs(newVal - u_curr[rowOffset + j]);
          if (diff > localMaxDiff) localMaxDiff = diff;
        }
      }

      // 4d. Right owned column (j = localCols) for interior rows i in [i_overlap_start, i_overlap_end]
      if (localCols > 1 && j_start <= localCols && localCols <= j_end && i_overlap_start <= i_overlap_end) {
        for (int i = i_overlap_start; i <= i_overlap_end; i++) {
          const size_t rowOffset = (size_t)i * LD;
          const size_t upOffset = (size_t)(i - 1) * LD;
          const size_t downOffset = (size_t)(i + 1) * LD;
          int j = localCols;

          double neighborsSum = u_curr[upOffset + j] + u_curr[downOffset + j] +
                                u_curr[rowOffset + j - 1] +
                                u_curr[rowOffset + j + 1];

          double newVal = (u_prev[rowOffset + j] + LAMBDA * neighborsSum) * invDiag;
          u_next[rowOffset + j] = newVal;

          double diff = std::abs(newVal - u_curr[rowOffset + j]);
          if (diff > localMaxDiff) localMaxDiff = diff;
        }
      }

      // Prepare for next Jacobi iteration
      u_curr.swap(u_next);
      local_comp_time += (MPI_Wtime() - t_comp1);

      double t_comm1 = MPI_Wtime();
      // Check global convergence across all MPI processes in 2D Cartesian topology
      double globalMaxDiff = 0.0;
      MPI_Allreduce(&localMaxDiff, &globalMaxDiff, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
      lastGlobalMaxDiff = globalMaxDiff;
      local_comm_time += (MPI_Wtime() - t_comm1);

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

    // Calculate maximum absolute residual r = b - A*u^{n+1} on the FINAL time step only
    if (t == STEPS - 1) {
      double t_comm_res = MPI_Wtime();
      MPI_Request res_reqs[8];
      MPI_Irecv(&u_curr[idx(0, 1)], localCols, MPI_DOUBLE, up, 1, cart_comm, &res_reqs[0]);
      MPI_Irecv(&u_curr[idx(localRows + 1, 1)], localCols, MPI_DOUBLE, down, 0, cart_comm, &res_reqs[1]);
      MPI_Isend(&u_curr[idx(1, 1)], localCols, MPI_DOUBLE, up, 0, cart_comm, &res_reqs[2]);
      MPI_Isend(&u_curr[idx(localRows, 1)], localCols, MPI_DOUBLE, down, 1, cart_comm, &res_reqs[3]);

      MPI_Irecv(&u_curr[idx(1, 0)], 1, col_type, left, 3, cart_comm, &res_reqs[4]);
      MPI_Irecv(&u_curr[idx(1, localCols + 1)], 1, col_type, right, 2, cart_comm, &res_reqs[5]);
      MPI_Isend(&u_curr[idx(1, 1)], 1, col_type, left, 2, cart_comm, &res_reqs[6]);
      MPI_Isend(&u_curr[idx(1, localCols)], 1, col_type, right, 3, cart_comm, &res_reqs[7]);

      MPI_Waitall(8, res_reqs, MPI_STATUSES_IGNORE);
      local_comm_time += (MPI_Wtime() - t_comm_res);

      double t_comp_res = MPI_Wtime();
      double stepLocalMaxResidual = 0.0;
      for (int i = i_start; i <= i_end; i++) {
        const size_t rowOffset = (size_t)i * LD;
        const size_t upOffset = (size_t)(i - 1) * LD;
        const size_t downOffset = (size_t)(i + 1) * LD;

        for (int j = j_start; j <= j_end; j++) {
          double Au = diag * u_curr[rowOffset + j] -
                      LAMBDA * (u_curr[upOffset + j] + u_curr[downOffset + j] +
                                u_curr[rowOffset + j - 1] + u_curr[rowOffset + j + 1]);
          double res = std::abs(u_prev[rowOffset + j] - Au);
          if (res > stepLocalMaxResidual) {
            stepLocalMaxResidual = res;
          }
        }
      }
      localMaxResidualOverall = stepLocalMaxResidual;
      local_comp_time += (MPI_Wtime() - t_comp_res);
    }

    // Advance time step: update RHS u_prev with converged u_curr,
    // leaving u_curr intact as the warm-start initial guess for time step t+1
    u_prev = u_curr;
  }

  MPI_Barrier(cart_comm);
  double end = MPI_Wtime();
  double localTime = end - start;
  double maxTime = 0.0;
  MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

  double maxCompTime = 0.0, maxCommTime = 0.0;
  MPI_Reduce(&local_comp_time, &maxCompTime, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
  MPI_Reduce(&local_comm_time, &maxCommTime, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

  double globalMaxResidual = 0.0;
  MPI_Reduce(&localMaxResidualOverall, &globalMaxResidual, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

  // Compute validation metrics (Max/Min temp, Total energy)
  double localMaxTemp = -1e9, localMinTemp = 1e9;
  double localTotalHeat = 0.0;
  for (int i = 1; i <= localRows; i++) {
    for (int j = 1; j <= localCols; j++) {
      double val = u_prev[idx(i, j)];
      if (val > localMaxTemp) localMaxTemp = val;
      if (val < localMinTemp) localMinTemp = val;
      localTotalHeat += val;
    }
  }

  double globalMaxTemp = 0.0, globalMinTemp = 0.0, globalTotalHeat = 0.0;
  MPI_Reduce(&localMaxTemp, &globalMaxTemp, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
  MPI_Reduce(&localMinTemp, &globalMinTemp, 1, MPI_DOUBLE, MPI_MIN, 0, cart_comm);
  MPI_Reduce(&localTotalHeat, &globalTotalHeat, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

  // Find center temperature for reporting
  double centerTemp = 0.0;
  if (center >= startRow && center < startRow + localRows &&
      center >= startCol && center < startCol + localCols) {
    centerTemp = u_prev[idx(center - startRow + 1, center - startCol + 1)];
  }
  double globalCenterTemp = 0.0;
  MPI_Reduce(&centerTemp, &globalCenterTemp, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

  if (rank == 0) {
    cout << fixed << setprecision(6);
    cout << "========================================================\n";
    cout << "Execution time       : " << fixed << setprecision(6) << maxTime << " seconds\n";
    double compPct = (maxTime > 0.0) ? (maxCompTime / maxTime * 100.0) : 0.0;
    double commPct = (maxTime > 0.0) ? (maxCommTime / maxTime * 100.0) : 0.0;
    cout << "  - Computation Time : " << fixed << setprecision(6) << maxCompTime << " seconds (" << fixed << setprecision(2) << compPct << "%)\n";
    cout << "  - Communication Time: " << fixed << setprecision(6) << maxCommTime << " seconds (" << fixed << setprecision(2) << commPct << "%)\n";
    cout << fixed << setprecision(6);
    if (STEPS > 0) {
      cout << "Time per time step   : " << (maxTime / STEPS) * 1000.0 << " ms\n";
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
    cout << "Max absolute residual: " << scientific << globalMaxResidual << defaultfloat << "\n";
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
         << " (Initial: 100.000000, Loss: " << (100.0 - globalTotalHeat) << ")\n";
    cout << "MPI processes        : " << size
         << " (2D Process Grid: " << dims[0] << " x " << dims[1] << ")\n";
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
  MPI_Bcast(&baselineExists, 1, MPI_INT, 0, cart_comm);

  if (baselineExists) {
    vector<double> ownedBlock(localRows * localCols);
    for (int i = 0; i < localRows; i++) {
      for (int j = 0; j < localCols; j++) {
        ownedBlock[i * localCols + j] = u_prev[idx(i + 1, j + 1)];
      }
    }

    vector<double> fullGrid;
    if (rank == 0) fullGrid.resize((size_t)N * N, 0.0);

    if (rank == 0) {
      for (int i = 0; i < localRows; i++) {
        for (int j = 0; j < localCols; j++) {
          fullGrid[(startRow + i) * N + (startCol + j)] = ownedBlock[i * localCols + j];
        }
      }
      for (int r = 1; r < size; r++) {
        int r_coords[2];
        MPI_Cart_coords(cart_comm, r, 2, r_coords);
        int r_rows = base_r + (r_coords[0] < rem_r ? 1 : 0);
        int r_startR = r_coords[0] * base_r + min(r_coords[0], rem_r);
        int r_cols = base_c + (r_coords[1] < rem_c ? 1 : 0);
        int r_startC = r_coords[1] * base_c + min(r_coords[1], rem_c);

        vector<double> r_buf(r_rows * r_cols);
        MPI_Recv(r_buf.data(), r_rows * r_cols, MPI_DOUBLE, r, 100, cart_comm, MPI_STATUS_IGNORE);
        for (int i = 0; i < r_rows; i++) {
          for (int j = 0; j < r_cols; j++) {
            fullGrid[(r_startR + i) * N + (r_startC + j)] = r_buf[i * r_cols + j];
          }
        }
      }

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
    } else {
      MPI_Send(ownedBlock.data(), localRows * localCols, MPI_DOUBLE, 0, 100, cart_comm);
    }
  }

  MPI_Type_free(&col_type);
  MPI_Comm_free(&cart_comm);
  MPI_Finalize();
  return 0;
}
