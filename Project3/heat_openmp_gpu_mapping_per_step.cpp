// Per-time-step data-mapping variant. The shared implementation keeps the
// same solver and kernel; GPU_MAPPING_POLICY controls only data-region scope.
#define GPU_MAPPING_POLICY 1
#include "heat_openmp_gpu_offload.cpp"
