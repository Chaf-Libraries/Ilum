#include "RHISampler.hpp"
#include "RHIDevice.hpp"

#include "Backend/DX12/Sampler.hpp"
#include "Backend/Vulkan/Sampler.hpp"
#ifdef CUDA_ENABLE
#	include "Backend/CUDA/Sampler.hpp"
#endif        // CUDA_ENABLE

namespace Ilum
{
SamplerDesc SamplerDesc::LinearClamp = SamplerDesc{
    RHIFilter::Linear,
    RHIFilter::Linear,
    RHIAddressMode::Clamp_To_Edge,
    RHIAddressMode::Clamp_To_Edge,
    RHIAddressMode::Clamp_To_Edge,
    RHIMipmapMode::Linear};

SamplerDesc SamplerDesc::LinearWarp = SamplerDesc{
    RHIFilter::Linear,
    RHIFilter::Linear,
    RHIAddressMode::Repeat,
    RHIAddressMode::Repeat,
    RHIAddressMode::Repeat,
    RHIMipmapMode::Linear};

SamplerDesc SamplerDesc::NearestClamp = SamplerDesc{
    RHIFilter::Nearest,
    RHIFilter::Nearest,
    RHIAddressMode::Clamp_To_Edge,
    RHIAddressMode::Clamp_To_Edge,
    RHIAddressMode::Clamp_To_Edge,
    RHIMipmapMode::Nearest};

SamplerDesc SamplerDesc::NearestWarp = SamplerDesc{
    RHIFilter::Nearest,
    RHIFilter::Nearest,
    RHIAddressMode::Repeat,
    RHIAddressMode::Repeat,
    RHIAddressMode::Repeat,
    RHIMipmapMode::Nearest};

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
#ifdef CUDA_ENABLE
		case RHIBackend::CUDA:
			return std::make_unique<CUDA::Sampler>(device, desc);
#endif
		default:
			break;
	}
	return nullptr;
}
}        // namespace Ilum