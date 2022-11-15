#include "RawCUDACompute.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <math.h>
#include <stdio.h>

namespace Ilum
{
__global__ void RawCUDAComputeKernal(cudaSurfaceObject_t surface)
{
	uint2 dispath_thread;
	dispath_thread.x = blockIdx.x * blockDim.x + threadIdx.x;
	dispath_thread.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (dispath_thread.x > 100 || dispath_thread.y > 100)
	{
		return;
	}

	float4 result = make_float4(0, 0, 0, 0);
	for (int i = 0; i < 0x00100000 >> 4; i++)
	{
		result.x = sin(result.x + float(threadIdx.x * threadIdx.x * (i * 4)));
		result.y = cos(result.y + float(threadIdx.y * threadIdx.y * (i * 4 + 1)));
	}

	result.w = 1.f;

	surf2Dwrite<float4>(result, surface, dispath_thread.x * sizeof(float4), dispath_thread.y, cudaBoundaryModeZero);
}

void ExecuteRawCUDAComputeKernal(cudaSurfaceObject_t surface, uint32_t thread_x, uint32_t thread_y, uint32_t thread_z, uint32_t block_x, uint32_t block_y, uint32_t block_z)
{
	RawCUDAComputeKernal<<<dim3((thread_x + block_x - 1) / block_x, (thread_y + block_y - 1) / block_y, (thread_z + block_z - 1) / block_z), dim3(block_x, block_y, block_z)>>>(surface);
}
}        // namespace Ilum