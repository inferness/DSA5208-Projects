#ifndef HEAT_JACOBI_KERNEL_H
#define HEAT_JACOBI_KERNEL_H

#include <cstddef>

double heat_jacobi_update(const double* u_prev, const double* u_curr,
                          double* u_next, std::size_t gridWidth,
                          int ownedRows, int firstGlobalRow,
                          int storageRowOffset, double lambda,
                          double invDiag);

#endif
