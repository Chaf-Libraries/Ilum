#include "RHISampler.hpp"
#include "RHIDevice.hpp"

#include "Backend/Vulkan/Sampler.hpp"
#include "Backend/DX12/Sampler.hpp"
#include "Backend/CUDA/Sampler.hpp"

namespace Ilum
{
RHISampler::RHISampler(RHIDevice *device, const SamplerDesc &desc) :
    p_device(device), m_desc(desc)
{
}

std::unique_ptr<RHISampler> RHISampler::Create(RHIDevice *device, const SamplerDesc &desc)
{
	switch (device->GetBackend())
	{
		case RHIBackend::Vulkan:
			return std::make_unique<Vulkan::Sampler>(device, desc);
		case RHIBackend::DX12:
			return std::make_unique<DX12::Sampler>(device, desc);
		case RHIBackend::CUDA:
			return std::make_unique<CUDA::Sampler>(device, desc);
		default:
			break;
	}
	return nullptr;
}
}        // namespace Ilum