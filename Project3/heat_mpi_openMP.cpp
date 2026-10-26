#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <vector>

using namespace std;

int main(int argc, char **argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (provided < MPI_THREAD_FUNNELED) {
    if (rank == 0) {
      cerr << "Error: MPI does not support the required thread level (MPI_THREAD_FUNNELED)\n";
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  const int N = 3000;
  const int STEPS = 1000;
  const double ALPHA = 0.05;

  if (N < 3 || STEPS < 0 || size > N) {
    if (rank == 0) {
      cerr << "Invalid arguments: N >= 3, "
           << "STEPS >= 0, processes <= N\n";
    }
    MPI_Finalize();
    return 1;
  }

  // Divide rows among MPI processes
  int base = N / size;
  int remainder = N % size;

  int localRows = base + (rank < remainder ? 1 : 0);
  int startRow = rank * base + min(rank, remainder);

  // Two extra rows for halo exchange
  vector<double> grid((localRows + 2) * N, 0.0);
  vector<double> next((localRows + 2) * N, 0.0);

  // Convert local row/column to 1D index
  auto idx = [N](int i, int j) { return i * N + j; };

  // Initialize the hot center
  int center = N / 2;
  if (center >= startRow && center < startRow + localRows) {
    int localCenter = center - startRow + 1;
    grid[idx(localCenter, center)] = 100.0;
  }

  int up = rank == 0 ? MPI_PROC_NULL : rank - 1;
  int down = rank == size - 1 ? MPI_PROC_NULL : rank + 1;

  // Determine row loop bounds (keeping global boundaries row 0 and row N-1 fixed at 0.0)
  int i_start = (startRow == 0) ? 2 : 1;
  int i_end = (startRow + localRows == N) ? localRows - 1 : localRows;

  int numThreads = 1;
  #pragma omp parallel
  {
    #pragma omp single
    numThreads = omp_get_num_threads();
  }

  double commTime = 0.0;
  double compTime = 0.0;
  double swapTime = 0.0;

  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  for (int t = 0; t < STEPS; t++) {

    // 1. Halo exchange (North / South)
    double t_comm_start = MPI_Wtime();
    // Send first owned row upward; receive lower neighbor into bottom halo
    MPI_Sendrecv(&grid[idx(1, 0)], N, MPI_DOUBLE, up, 0,
                 &grid[idx(localRows + 1, 0)], N, MPI_DOUBLE, down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send last owned row downward; receive upper neighbor into top halo
    MPI_Sendrecv(&grid[idx(localRows, 0)], N, MPI_DOUBLE, down, 1,
                 &grid[idx(0, 0)], N, MPI_DOUBLE, up, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    commTime += (MPI_Wtime() - t_comm_start);

    // 2. Multi-threaded interior stencil computation using OpenMP
    double t_comp_start = MPI_Wtime();
    #pragma omp parallel for schedule(static)
    for (int i = i_start; i <= i_end; i++) {
      for (int j = 1; j < N - 1; j++) {
        next[idx(i, j)] = grid[idx(i, j)] +
                          ALPHA * (grid[idx(i - 1, j)] + grid[idx(i + 1, j)] +
                                   grid[idx(i, j - 1)] + grid[idx(i, j + 1)] -
                                   4.0 * grid[idx(i, j)]);
      }
    }
    compTime += (MPI_Wtime() - t_comp_start);

    // 3. Grid swap
    double t_swap_start = MPI_Wtime();
    grid.swap(next);
    swapTime += (MPI_Wtime() - t_swap_start);
  }

  double localTime = MPI_Wtime() - start;

  // Max across all ranks
  double maxTotalTime = 0.0;
  double maxComm = 0.0, maxComp = 0.0, maxSwap = 0.0;

  MPI_Reduce(&localTime, &maxTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&commTime, &maxComm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&compTime, &maxComp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&swapTime, &maxSwap, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // Average across all ranks
  double sumComm = 0.0, sumComp = 0.0, sumSwap = 0.0;
  MPI_Reduce(&commTime, &sumComm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&compTime, &sumComp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&swapTime, &sumSwap, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    double avgComm = sumComm / size;
    double avgComp = sumComp / size;
    double avgSwap = sumSwap / size;

    cout << fixed << setprecision(6);
    cout << "========================================================\n";
    cout << "Hybrid MPI + OpenMP Profiling Results (heat_mpi_openMP)\n";
    cout << "========================================================\n";
    cout << "Grid:                        " << N << " x " << N << "\n";
    cout << "Steps:                       " << STEPS << "\n";
    cout << "MPI processes:               " << size << "\n";
    cout << "OpenMP threads per process:  " << numThreads << "\n";
    cout << "Total workers (MPI x OMP):   " << size * numThreads << "\n";
    cout << "--------------------------------------------------------\n";
    cout << "Total Execution Time:        " << setw(10) << maxTotalTime << " s (100.0%)\n";
    cout << "--------------------------------------------------------\n";
    cout << "Breakdown (Average across ranks):\n";
    cout << "  - Stencil Computation:     " << setw(10) << avgComp
         << " s (" << setw(5) << setprecision(2) << (avgComp / maxTotalTime * 100.0) << "%)\n";
    cout << fixed << setprecision(6);
    cout << "  - Halo Communication:      " << setw(10) << avgComm
         << " s (" << setw(5) << setprecision(2) << (avgComm / maxTotalTime * 100.0) << "%)\n";
    cout << fixed << setprecision(6);
    cout << "  - Grid Swap:               " << setw(10) << avgSwap
         << " s (" << setw(5) << setprecision(2) << (avgSwap / maxTotalTime * 100.0) << "%)\n";
    cout << "--------------------------------------------------------\n";
    cout << fixed << setprecision(6);
    cout << "Breakdown (Max across ranks):\n";
    cout << "  - Stencil Computation:     " << setw(10) << maxComp << " s\n";
    cout << "  - Halo Communication:      " << setw(10) << maxComm << " s\n";
    cout << "  - Grid Swap:               " << setw(10) << maxSwap << " s\n";
    cout << "========================================================\n";
  }

  MPI_Finalize();
  return 0;
}
