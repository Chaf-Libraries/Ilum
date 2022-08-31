#include "Device.hpp"

#include <cuda_runtime.h>

namespace Ilum::CUDA
{
Device::Device()
{
	int32_t device_count = 0;
	cudaGetDeviceCount(&device_count);
	for (int32_t i = 0; i < device_count; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		std::stringstream ss;
		ss << "\nFound physical device [" << i << "]\n";
		ss << "Name: " << prop.name << "\n";
		ss << "Vendor: ";
		ss << "Nvidia\n";
		LOG_INFO("{}", ss.str());
	}

	cuInit(0);
	cuDeviceGet(&m_device, 0);
	cuCtxCreate(&m_context, 0, m_device);
}

Device::~Device()
{
	cuCtxDestroy(m_context);
}

void Device::WaitIdle()
{
	cudaDeviceSynchronize();
}

bool Device::IsFeatureSupport(RHIFeature feature)
{
	return false;
}
}        // namespace Ilum::CUDA