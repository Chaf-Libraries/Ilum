#include "RHIDevice.hpp"

#include "Backend/DX12/Device.hpp"
#include "Backend/Vulkan/Device.hpp"
#ifdef CUDA_ENABLE
#	include "Backend/CUDA/Device.hpp"
#endif        // CUDA_ENABLE

namespace Ilum
{
RHIDevice::RHIDevice(RHIBackend backend) :
    m_backend(backend)
{
}

std::unique_ptr<RHIDevice> RHIDevice::Create(RHIBackend backend)
{
	switch (backend)
	{
		case RHIBackend::Vulkan:
			return std::make_unique<Vulkan::Device>();
		case RHIBackend::DX12:
			return std::make_unique<DX12::Device>();
#ifdef CUDA_ENABLE
		case RHIBackend::CUDA:
			return std::make_unique<CUDA::Device>();
#endif        // CUDA_ENABLE
		default:
			break;
	}
	return nullptr;
}

const std::string &RHIDevice::GetName() const
{
	return m_name;
}

RHIBackend RHIDevice::GetBackend() const
{
	return m_backend;
}
}        // namespace Ilum