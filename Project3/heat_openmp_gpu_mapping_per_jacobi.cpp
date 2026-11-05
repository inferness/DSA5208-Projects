// Per-Jacobi-iteration data-mapping variant. With no enclosing target-data
// region, each kernel's map clauses transfer its input/output grids.
#define GPU_MAPPING_POLICY 2
#include "heat_openmp_gpu_offload.cpp"
