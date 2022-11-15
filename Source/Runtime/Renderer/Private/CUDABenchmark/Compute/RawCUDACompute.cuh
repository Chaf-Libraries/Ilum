#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace Ilum
{
void ExecuteRawCUDAComputeKernal(cudaSurfaceObject_t surface, uint32_t thread_x, uint32_t thread_y, uint32_t thread_z, uint32_t block_x, uint32_t block_y, uint32_t block_z);
}