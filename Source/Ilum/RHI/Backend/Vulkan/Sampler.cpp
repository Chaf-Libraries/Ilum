#include "Sampler.hpp"
#include "Definitions.hpp"
#include "Device.hpp"

namespace Ilum::Vulkan
{
Sampler::Sampler(RHIDevice *device, const SamplerDesc &desc) :
    RHISampler(device, desc)
{
	VkSamplerCreateInfo create_info = {};
	create_info.sType               = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	create_info.minFilter           = ToVulkanFilter[desc.min_filter];
	create_info.magFilter           = ToVulkanFilter[desc.mag_filter];
	create_info.mipmapMode          = ToVulkanMipmapMode[desc.mipmap_mode];
	create_info.addressModeU        = ToVulkanAddressMode[desc.address_mode_u];
	create_info.addressModeV        = ToVulkanAddressMode[desc.address_mode_v];
	create_info.addressModeW        = ToVulkanAddressMode[desc.address_mode_w];
	create_info.anisotropyEnable    = desc.anisotropic;
	create_info.maxAnisotropy       = 1000.f;
	create_info.mipLodBias          = desc.mip_lod_bias;
	create_info.minLod              = desc.min_lod;
	create_info.maxLod              = desc.max_lod;
	create_info.borderColor         = ToVulkanBorderColor[desc.border_color];

	vkCreateSampler(static_cast<Device *>(p_device)->GetDevice(), &create_info, nullptr, &m_handle);
}

Sampler::~Sampler()
{
	if (m_handle)
	{
		vkDestroySampler(static_cast<Device *>(p_device)->GetDevice(), m_handle, nullptr);
	}
}

VkSampler Sampler::GetHandle() const
{
	return m_handle;
}
}        // namespace Ilum::Vulkan