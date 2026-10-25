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

  const int N = 10000;
  const int STEPS = 1000;
  const double ALPHA = 0.05;
  const int TILE_SIZE = 64;

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

  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  for (int t = 0; t < STEPS; t++) {

    // Send first owned row upward;
    // receive lower neighbor into bottom halo.
    MPI_Sendrecv(&grid[idx(1, 0)], N, MPI_DOUBLE, up, 0,
                 &grid[idx(localRows + 1, 0)], N, MPI_DOUBLE, down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send last owned row downward;
    // receive upper neighbor into top halo.
    MPI_Sendrecv(&grid[idx(localRows, 0)], N, MPI_DOUBLE, down, 1,
                 &grid[idx(0, 0)], N, MPI_DOUBLE, up, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    // Update owned interior cells using 2D spatial tiling (32x32)
    for (int ii = 1; ii <= localRows; ii += TILE_SIZE) {
      int i_end = min(ii + TILE_SIZE, localRows + 1);

      for (int jj = 1; jj < N - 1; jj += TILE_SIZE) {
        int j_end = min(jj + TILE_SIZE, N - 1);

        for (int i = ii; i < i_end; i++) {
          int globalRow = startRow + i - 1;

          // Keep top and bottom boundaries at 0
          if (globalRow == 0 || globalRow == N - 1)
            continue;

          for (int j = jj; j < j_end; j++) {
            next[idx(i, j)] =
                grid[idx(i, j)] +
                ALPHA * (grid[idx(i - 1, j)] + grid[idx(i + 1, j)] +
                         grid[idx(i, j - 1)] + grid[idx(i, j + 1)] -
                         4.0 * grid[idx(i, j)]);
          }
        }
      }
    }

    grid.swap(next);
  }

  double localTime = MPI_Wtime() - start;
  double maxTime = 0.0;

  MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    cout << fixed << setprecision(6);
    cout << "Grid: " << N << " x " << N << "\n";
    cout << "Steps: " << STEPS << "\n";
    cout << "Tile size: " << TILE_SIZE << " x " << TILE_SIZE << "\n";
    cout << "MPI processes: " << size << "\n";
    cout << "Execution time: " << maxTime << " seconds\n";
  }

  MPI_Finalize();
  return 0;
}
