#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace Ilum
{
void ExecuteRawCUDATextureKernal(cudaSurfaceObject_t surface, cudaTextureObject_t texture, uint32_t thread_x, uint32_t thread_y, uint32_t thread_z, uint32_t block_x, uint32_t block_y, uint32_t block_z);
}