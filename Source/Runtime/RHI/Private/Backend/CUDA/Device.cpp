#include "Device.hpp"

#include <cuda_runtime.h>

namespace Ilum::CUDA
{
Device::Device() :
    RHIDevice(RHIBackend::CUDA)
{
	LOG_INFO("Initializing RHI backend CUDA...");

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	m_name = prop.name;

	cuInit(0);
	cuDeviceGet(&m_device, 0);
	cuCtxCreate(&m_context, 0, m_device);
	cudaStreamCreate(&m_steam);
}

Device::~Device()
{
	cuCtxDestroy(m_context);
}

void Device::WaitIdle()
{
	cudaStreamSynchronize(m_steam);
}

bool Device::IsFeatureSupport(RHIFeature feature)
{
	return false;
}

cudaStream_t Device::GetSteam() const
{
	return m_steam;
}
}        // namespace Ilum::CUDA