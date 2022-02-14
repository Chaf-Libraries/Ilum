#include "Sampler.hpp"
#include "../Device/Device.hpp"

namespace Ilum::Graphics
{
Sampler::Sampler(const Device &device) :
    m_device(device)
{
}

Sampler::Sampler(const Device &device, VkFilter min_filter, VkFilter mag_filter, VkSamplerAddressMode address_mode, VkFilter mip_filter) :
    m_device(device)
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

	auto result = vkCreateSampler(m_device, &create_info, nullptr, &m_handle);
}

Sampler::~Sampler()
{
	if (m_handle)
	{
		vkDestroySampler(m_device, m_handle, nullptr);
	}
}

Sampler::Sampler(Sampler &&other) noexcept:
    m_device(other.m_device),
    m_handle(other.m_handle)
{
	other.m_handle = VK_NULL_HANDLE;
}

Sampler &Sampler::operator=(Sampler &&other) noexcept
{
	m_handle = other.m_handle;
	other.m_handle = VK_NULL_HANDLE;
	return *this;
}

Sampler::operator const VkSampler &() const
{
	return m_handle;
}

const VkSampler &Sampler::GetHandle() const
{
	return m_handle;
}
}        // namespace Ilum::Graphics