#include "Sampler.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Device/LogicalDevice.hpp"

namespace Ilum
{
Sampler::Sampler(VkSampler sampler):
    m_handle(sampler)
{
}

Sampler::Sampler(VkFilter min_filter, VkFilter mag_filter, VkSamplerAddressMode address_mode, VkFilter mip_filter)
{
	VkSamplerCreateInfo create_info = {};
	create_info.sType               = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	create_info.minFilter           = min_filter;
	create_info.magFilter           = mag_filter;
	create_info.addressModeU        = address_mode;
	create_info.addressModeV        = address_mode;
	create_info.addressModeW        = address_mode;
	create_info.mipLodBias          = 0;
	create_info.minLod              = 0;
	create_info.maxLod              = 1000;

	auto result = vkCreateSampler(GraphicsContext::instance()->getLogicalDevice(), &create_info, nullptr, &m_handle);
}

Sampler::~Sampler()
{
	if (m_handle)
	{
		vkDestroySampler(GraphicsContext::instance()->getLogicalDevice(), m_handle, nullptr);
	}
}

Sampler::Sampler(Sampler &&other):
    m_handle(other.m_handle)
{
	other.m_handle = VK_NULL_HANDLE;
}

Sampler &Sampler::operator=(Sampler &&other)
{
	m_handle       = other.m_handle;
	other.m_handle = VK_NULL_HANDLE;

	return *this;
}

const VkSampler &Sampler::getSampler() const
{
	return m_handle;
}

Sampler::operator const VkSampler &() const
{
	return m_handle;
}
}        // namespace Ilum