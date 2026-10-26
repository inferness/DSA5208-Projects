#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <vector>

using namespace std;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int N = 3000;
  const int STEPS = 1000;
  const double ALPHA = 0.05;

  if (N < 3 || STEPS < 0) {
    if (rank == 0) {
      cerr << "Invalid arguments: N >= 3, STEPS >= 0\n";
    }
    MPI_Finalize();
    return 1;
  }

  // Create 2D Cartesian process grid
  int dims[2] = {0, 0};
  MPI_Dims_create(size, 2, dims);

  int periods[2] = {0, 0}; // Non-periodic domain
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

  int coords[2];
  MPI_Cart_coords(cart_comm, rank, 2, coords);

  // Determine neighboring ranks (returns MPI_PROC_NULL on boundaries)
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

  // Local grid dimensions with halo layer (1 cell on each of the 4 borders)
  int LD = localCols + 2;
  vector<double> grid((localRows + 2) * (localCols + 2), 0.0);
  vector<double> next((localRows + 2) * (localCols + 2), 0.0);

  // Convert local 2D index (i in [0..localRows+1], j in [0..localCols+1]) to 1D index
  auto idx = [LD](int i, int j) { return i * LD + j; };

  // Initialize the hot center if this rank owns it
  int center = N / 2;
  if (center >= startRow && center < startRow + localRows &&
      center >= startCol && center < startCol + localCols) {
    int localCenterRow = center - startRow + 1;
    int localCenterCol = center - startCol + 1;
    grid[idx(localCenterRow, localCenterCol)] = 100.0;
  }

  // Define derived datatype for sending/receiving non-contiguous vertical column halos
  MPI_Datatype col_type;
  MPI_Type_vector(localRows, 1, LD, MPI_DOUBLE, &col_type);
  MPI_Type_commit(&col_type);

  // Loop bounds for interior computation (keep global boundaries at 0)
  int i_start = (startRow == 0) ? 2 : 1;
  int i_end = (startRow + localRows == N) ? localRows - 1 : localRows;
  int j_start = (startCol == 0) ? 2 : 1;
  int j_end = (startCol + localCols == N) ? localCols - 1 : localCols;

  double commRowTime = 0.0;
  double commColTime = 0.0;
  double compTime = 0.0;
  double swapTime = 0.0;

  MPI_Barrier(cart_comm);
  double start = MPI_Wtime();

  for (int t = 0; t < STEPS; t++) {

    // 1. Halo exchange North / South (Up / Down) - contiguous rows
    double t_row_start = MPI_Wtime();
    // Send row 1 upward, receive into bottom halo
    MPI_Sendrecv(&grid[idx(1, 1)], localCols, MPI_DOUBLE, up, 0,
                 &grid[idx(localRows + 1, 1)], localCols, MPI_DOUBLE, down, 0,
                 cart_comm, MPI_STATUS_IGNORE);

    // Send row localRows downward, receive into top halo
    MPI_Sendrecv(&grid[idx(localRows, 1)], localCols, MPI_DOUBLE, down, 1,
                 &grid[idx(0, 1)], localCols, MPI_DOUBLE, up, 1,
                 cart_comm, MPI_STATUS_IGNORE);
    commRowTime += (MPI_Wtime() - t_row_start);

    // 2. Halo exchange West / East (Left / Right) - non-contiguous columns
    double t_col_start = MPI_Wtime();
    // Send col 1 to left, receive into right halo
    MPI_Sendrecv(&grid[idx(1, 1)], 1, col_type, left, 2,
                 &grid[idx(1, localCols + 1)], 1, col_type, right, 2,
                 cart_comm, MPI_STATUS_IGNORE);

    // Send col localCols to right, receive into left halo
    MPI_Sendrecv(&grid[idx(1, localCols)], 1, col_type, right, 3,
                 &grid[idx(1, 0)], 1, col_type, left, 3,
                 cart_comm, MPI_STATUS_IGNORE);
    commColTime += (MPI_Wtime() - t_col_start);

    // 3. Update owned interior cells (computation)
    double t_comp_start = MPI_Wtime();
    for (int i = i_start; i <= i_end; i++) {
      for (int j = j_start; j <= j_end; j++) {
        next[idx(i, j)] = grid[idx(i, j)] +
                          ALPHA * (grid[idx(i - 1, j)] + grid[idx(i + 1, j)] +
                                   grid[idx(i, j - 1)] + grid[idx(i, j + 1)] -
                                   4.0 * grid[idx(i, j)]);
      }
    }
    compTime += (MPI_Wtime() - t_comp_start);

    // 4. Grid swap
    double t_swap_start = MPI_Wtime();
    grid.swap(next);
    swapTime += (MPI_Wtime() - t_swap_start);
  }

  double localTime = MPI_Wtime() - start;

  // Max across all ranks
  double maxTotalTime = 0.0;
  double maxCommRow = 0.0, maxCommCol = 0.0, maxComp = 0.0, maxSwap = 0.0;

  MPI_Reduce(&localTime, &maxTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
  MPI_Reduce(&commRowTime, &maxCommRow, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
  MPI_Reduce(&commColTime, &maxCommCol, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
  MPI_Reduce(&compTime, &maxComp, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
  MPI_Reduce(&swapTime, &maxSwap, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

  // Average across all ranks
  double sumCommRow = 0.0, sumCommCol = 0.0, sumComp = 0.0, sumSwap = 0.0;
  MPI_Reduce(&commRowTime, &sumCommRow, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
  MPI_Reduce(&commColTime, &sumCommCol, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
  MPI_Reduce(&compTime, &sumComp, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
  MPI_Reduce(&swapTime, &sumSwap, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

  if (rank == 0) {
    double avgCommRow = sumCommRow / size;
    double avgCommCol = sumCommCol / size;
    double avgComp = sumComp / size;
    double avgSwap = sumSwap / size;
    double totalComm = avgCommRow + avgCommCol;

    cout << fixed << setprecision(6);
    cout << "========================================================\n";
    cout << "2D Block Decomposition Profiling Results\n";
    cout << "========================================================\n";
    cout << "Grid: " << N << " x " << N << "\n";
    cout << "Steps: " << STEPS << "\n";
    cout << "MPI processes: " << size << " (2D Grid: " << dims[0] << " x " << dims[1] << ")\n";
    cout << "--------------------------------------------------------\n";
    cout << "Total Execution Time:        " << setw(10) << maxTotalTime << " s (100.0%)\n";
    cout << "--------------------------------------------------------\n";
    cout << "Breakdown (Average across ranks):\n";
    cout << "  - Stencil Computation:     " << setw(10) << avgComp
         << " s (" << setw(5) << setprecision(2) << (avgComp / maxTotalTime * 100.0) << "%)\n";
    cout << fixed << setprecision(6);
    cout << "  - Total Communication:     " << setw(10) << totalComm
         << " s (" << setw(5) << setprecision(2) << (totalComm / maxTotalTime * 100.0) << "%)\n";
    cout << fixed << setprecision(6);
    cout << "      * North/South (Rows):  " << setw(10) << avgCommRow
         << " s (" << setw(5) << setprecision(2) << (avgCommRow / maxTotalTime * 100.0) << "%)\n";
    cout << fixed << setprecision(6);
    cout << "      * West/East (Columns): " << setw(10) << avgCommCol
         << " s (" << setw(5) << setprecision(2) << (avgCommCol / maxTotalTime * 100.0) << "%)\n";
    cout << fixed << setprecision(6);
    cout << "  - Grid Swap:               " << setw(10) << avgSwap
         << " s (" << setw(5) << setprecision(2) << (avgSwap / maxTotalTime * 100.0) << "%)\n";
    cout << "--------------------------------------------------------\n";
    cout << fixed << setprecision(6);
    cout << "Breakdown (Max across ranks):\n";
    cout << "  - Stencil Computation:     " << setw(10) << maxComp << " s\n";
    cout << "  - North/South (Rows):      " << setw(10) << maxCommRow << " s\n";
    cout << "  - West/East (Columns):     " << setw(10) << maxCommCol << " s\n";
    cout << "  - Grid Swap:               " << setw(10) << maxSwap << " s\n";
    cout << "========================================================\n";
  }

  MPI_Type_free(&col_type);
  MPI_Comm_free(&cart_comm);
  MPI_Finalize();
  return 0;
}
