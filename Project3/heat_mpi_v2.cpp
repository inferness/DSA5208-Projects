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

  MPI_Request reqs[4];

  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  for (int t = 0; t < STEPS; t++) {
    // 1. Post non-blocking receives into halo buffers
    MPI_Irecv(&grid[idx(0, 0)], N, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(&grid[idx(localRows + 1, 0)], N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &reqs[1]);

    // 2. Post non-blocking sends of boundary rows
    MPI_Isend(&grid[idx(1, 0)], N, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(&grid[idx(localRows, 0)], N, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &reqs[3]);

    // 3. Compute interior rows (independent of halos) to overlap communication with computation
    for (int i = 2; i <= localRows - 1; i++) {
      int globalRow = startRow + i - 1;
      if (globalRow == 0 || globalRow == N - 1)
        continue;

      for (int j = 1; j < N - 1; j++) {
        next[idx(i, j)] = grid[idx(i, j)] +
                          ALPHA * (grid[idx(i - 1, j)] + grid[idx(i + 1, j)] +
                                   grid[idx(i, j - 1)] + grid[idx(i, j + 1)] -
                                   4.0 * grid[idx(i, j)]);
      }
    }

    // 4. Wait for all halo communications to complete
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    // 5. Compute top boundary row (i = 1)
    if (localRows >= 1) {
      int globalRow = startRow;
      if (globalRow != 0 && globalRow != N - 1) {
        for (int j = 1; j < N - 1; j++) {
          next[idx(1, j)] = grid[idx(1, j)] +
                            ALPHA * (grid[idx(0, j)] + grid[idx(2, j)] +
                                     grid[idx(1, j - 1)] + grid[idx(1, j + 1)] -
                                     4.0 * grid[idx(1, j)]);
        }
      }
    }

    // 6. Compute bottom boundary row (i = localRows) if distinct from top row
    if (localRows > 1) {
      int globalRow = startRow + localRows - 1;
      if (globalRow != 0 && globalRow != N - 1) {
        for (int j = 1; j < N - 1; j++) {
          next[idx(localRows, j)] = grid[idx(localRows, j)] +
                            ALPHA * (grid[idx(localRows - 1, j)] + grid[idx(localRows + 1, j)] +
                                     grid[idx(localRows, j - 1)] + grid[idx(localRows, j + 1)] -
                                     4.0 * grid[idx(localRows, j)]);
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
    cout << "MPI processes: " << size << "\n";
    cout << "Execution time: " << maxTime << " seconds\n";
  }

  MPI_Finalize();
  return 0;
}
