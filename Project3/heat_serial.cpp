
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  const int N = 10000;
  const int STEPS = 1000;
  const double ALPHA = 0.05;

  // Two grids: current and next time step
  vector<vector<double>> grid(N, vector<double>(N, 0.0));
  vector<vector<double>> next(N, vector<double>(N, 0.0));

  // Initial condition: hot cell in the center
  grid[N / 2][N / 2] = 100.0;

  auto start = chrono::steady_clock::now();

  for (int t = 0; t < STEPS; t++) {
    // Update interior cells
    for (int i = 1; i < N - 1; i++) {
      for (int j = 1; j < N - 1; j++) {
        next[i][j] = grid[i][j] +
                     ALPHA * (grid[i - 1][j] + grid[i + 1][j] + grid[i][j - 1] +
                              grid[i][j + 1] - 4.0 * grid[i][j]);
      }
    }

    // Swap grids for the next iteration
    grid.swap(next);
  }

  auto end = chrono::steady_clock::now();

  double elapsed = chrono::duration<double>(end - start).count();

  cout << fixed << setprecision(6);
  cout << "Grid size: " << N << " x " << N << "\n";
  cout << "Time steps: " << STEPS << "\n";
  cout << "Execution time: " << elapsed << " seconds\n";
  cout << "Center temperature: " << grid[N / 2][N / 2] << "\n";

  return 0;
}
