#include "Sampler.hpp"
#include "Device.hpp"

namespace Ilum
{
Sampler::Sampler(RHIDevice *device, const SamplerDesc &desc):
    p_device(device), m_desc(desc)
{
	VkSamplerCreateInfo create_info = {};
	create_info.sType               = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	create_info.minFilter           = desc.min_filter;
	create_info.magFilter           = desc.mag_filter;
	create_info.mipmapMode          = desc.mipmap_mode;
	create_info.addressModeU        = desc.address_mode;
	create_info.addressModeV        = desc.address_mode;
	create_info.addressModeW        = desc.address_mode;
	create_info.anisotropyEnable    = desc.anisotropic;
	create_info.maxAnisotropy       = 16.f;
	create_info.mipLodBias          = desc.mip_lod_bias;
	create_info.minLod              = desc.min_lod;
	create_info.maxLod              = desc.max_lod;

	vkCreateSampler(p_device->GetDevice(), &create_info, nullptr, &m_handle);
}

Sampler::~Sampler()
{
	if (m_handle)
	{
		p_device->WaitIdle();
		vkDestroySampler(p_device->GetDevice(), m_handle, nullptr);
	}
}

Sampler::operator VkSampler() const
{
	return m_handle;
}
}        // namespace Ilum