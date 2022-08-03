#include "RHISampler.hpp"

#ifdef RHI_BACKEND_VULKAN
#	include "Backend/Vulkan/Sampler.hpp"
#elif defined RHI_BACKEND_DX12
#	include "Backend/DX12/Sampler.hpp"
#endif        // RHI_BACKEND

namespace Ilum
{
RHISampler::RHISampler(RHIDevice *device, const SamplerDesc &desc):
    p_device(device), m_desc(desc)
{
}

std::unique_ptr<RHISampler> RHISampler::Create(RHIDevice *device, const SamplerDesc &desc)
{
#ifdef RHI_BACKEND_VULKAN
	return std::make_unique<Vulkan::Sampler>(device, desc);
#elif defined RHI_BACKEND_DX12
	return std::make_unique<DX12::Sampler>(device, desc);
#else
	return nullptr;
#endif        // RHI_BACKEND
}
}        // namespace Ilum