#include "heat_jacobi_kernel.h"

#include <cmath>

#if defined(__GNUC__) || defined(__clang__)
#define HEAT_UNLIKELY(condition) __builtin_expect(!!(condition), 0)
#else
#define HEAT_UNLIKELY(condition) (condition)
#endif

double heat_jacobi_update(const double* u_prev, const double* u_curr,
                          double* u_next, std::size_t gridWidth,
                          int ownedRows, int firstGlobalRow,
                          int storageRowOffset, double lambda,
                          double invDiag) {
  double maxDiff = 0.0;
  const int gridRows = static_cast<int>(gridWidth);

  for (int localRow = 0; localRow < ownedRows; ++localRow) {
    const int globalRow = firstGlobalRow + localRow;
    if (globalRow == 0 || globalRow == gridRows - 1) {
      continue;
    }

    const std::size_t row =
        static_cast<std::size_t>(localRow + storageRowOffset) * gridWidth;
    const std::size_t up = row - gridWidth;
    const std::size_t down = row + gridWidth;

    for (std::size_t col = 1; col < gridWidth - 1; ++col) {
      const double neighborsSum = u_curr[up + col] + u_curr[down + col] +
                                  u_curr[row + col - 1] +
                                  u_curr[row + col + 1];
      const double newVal =
          (u_prev[row + col] + lambda * neighborsSum) * invDiag;
      u_next[row + col] = newVal;

      const double diff = std::abs(newVal - u_curr[row + col]);
      if (HEAT_UNLIKELY(diff > maxDiff)) {
        maxDiff = diff;
      }
    }
  }

  return maxDiff;
}

#undef HEAT_UNLIKELY
