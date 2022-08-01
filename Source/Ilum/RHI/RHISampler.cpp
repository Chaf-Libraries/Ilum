#include "RHISampler.hpp"

#include"Vulkan/Sampler.hpp"

namespace Ilum
{
RHISampler::RHISampler(RHIDevice *device, const SamplerDesc &desc):
    p_device(device), m_desc(desc)
{
}

std::unique_ptr<RHISampler> RHISampler::Create(RHIDevice *device, const SamplerDesc &desc)
{
	return std::make_unique<Vulkan::Sampler>(device, desc);
}
}        // namespace Ilum