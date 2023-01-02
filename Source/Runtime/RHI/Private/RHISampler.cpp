#include "RHISampler.hpp"
#include "RHIDevice.hpp"

#include <Core/Plugin.hpp>

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
	return std::unique_ptr<RHISampler>(std::move(PluginManager::GetInstance().Call<RHISampler *>(fmt::format("shared/RHI/RHI.{}.dll", device->GetBackend()), "CreateSampler", device, desc)));
}
}        // namespace Ilum