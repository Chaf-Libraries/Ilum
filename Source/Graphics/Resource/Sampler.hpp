#pragma once

#include "Graphics/Vulkan.hpp"

namespace Ilum::Graphics
{
class Device;

class Sampler
{
  public:
	Sampler(const Device &device);
	Sampler(const Device &device, VkFilter min_filter, VkFilter mag_filter, VkSamplerAddressMode address_mode, VkFilter mip_filter);
	~Sampler();

	Sampler(Sampler &&other) noexcept;
	Sampler &operator        =(Sampler &&other) noexcept;
	Sampler(const Sampler &)            = delete;
	Sampler &operator=(const Sampler &other) = delete;

	operator const VkSampler &() const;

	const VkSampler &GetHandle() const;

  private:
	const Device &m_device;
	VkSampler     m_handle = VK_NULL_HANDLE;
};

using SamplerReference = std::reference_wrapper<const Sampler>;
}        // namespace Ilum::Graphics