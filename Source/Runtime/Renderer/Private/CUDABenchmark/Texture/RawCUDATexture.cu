#include "RawCUDATexture.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <math.h>
#include <stdio.h>

namespace Ilum
{
__device__ float4 operator*(float a, float4 b)
{
	float4 r;
	r.x = a * b.x;
	r.y = a * b.y;
	r.z = a * b.z;
	r.w = a * b.w;
	return r;
}

__device__ float4 operator+(float4 a, float4 b)
{
	float4 r;
	r.x = a.x + b.x;
	r.y = a.y + b.y;
	r.z = a.z + b.z;
	r.w = a.w + b.w;
	return r;
}

__device__ float4 operator-(float4 a, float4 b)
{
	float4 r;
	r.x = a.x - b.x;
	r.y = a.y - b.y;
	r.z = a.z - b.z;
	r.w = a.w - b.w;
	return r;
}

__global__ void RawCUDATextureKernal(cudaSurfaceObject_t surface, cudaTextureObject_t texture)
{
	uint2 dispath_thread;
	dispath_thread.x = blockIdx.x * blockDim.x + threadIdx.x;
	dispath_thread.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (dispath_thread.x > 100 || dispath_thread.y > 100)
	{
		return;
	}

	float4 x00 = tex2DLod<float4>(texture, float(dispath_thread.x - 1) / 100.f, float(dispath_thread.y - 1) / 100.f, 0.0);
	float4 x01 = tex2DLod<float4>(texture, float(dispath_thread.x - 1) / 100.f, float(dispath_thread.y) / 100.f, 0.0);
	float4 x02 = tex2DLod<float4>(texture, float(dispath_thread.x - 1) / 100.f, float(dispath_thread.y + 1) / 100.f, 0.0);
	float4 x10 = tex2DLod<float4>(texture, float(dispath_thread.x) / 100.f, float(dispath_thread.y - 1) / 100.f, 0.0);
	float4 x11 = tex2DLod<float4>(texture, float(dispath_thread.x) / 100.f, float(dispath_thread.y) / 100.f, 0.0);
	float4 x12 = tex2DLod<float4>(texture, float(dispath_thread.x) / 100.f, float(dispath_thread.y + 1) / 100.f, 0.0);
	float4 x20 = tex2DLod<float4>(texture, float(dispath_thread.x + 1) / 100.f, float(dispath_thread.y - 1) / 100.f, 0.0);
	float4 x21 = tex2DLod<float4>(texture, float(dispath_thread.x + 1) / 100.f, float(dispath_thread.y) / 100.f, 0.0);
	float4 x22 = tex2DLod<float4>(texture, float(dispath_thread.x + 1) / 100.f, float(dispath_thread.y + 1) / 100.f, 0.0);

	float4 result = 1.f / 12.f * (x00 + 2 * x01 + x02 + 2 * x10 - 12 * x11 + 2 * x12 + x20 + 2 * x21 + x22);
	result.w      = 1.f;
	surf2Dwrite<float4>(result, surface, dispath_thread.x * sizeof(float4), dispath_thread.y, cudaBoundaryModeZero);
}

void ExecuteRawCUDATextureKernal(cudaSurfaceObject_t surface, cudaTextureObject_t texture, uint32_t thread_x, uint32_t thread_y, uint32_t thread_z, uint32_t block_x, uint32_t block_y, uint32_t block_z)
{
	RawCUDATextureKernal<<<dim3((thread_x + block_x - 1) / block_x, (thread_y + block_y - 1) / block_y, (thread_z + block_z - 1) / block_z), dim3(block_x, block_y, block_z)>>>(surface, texture);
}
}        // namespace Ilum