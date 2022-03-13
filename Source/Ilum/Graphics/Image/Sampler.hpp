#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class Sampler
{
  public:
	Sampler() = default;

	Sampler(VkSampler sampler);

	Sampler(VkFilter min_filter, VkFilter mag_filter, VkSamplerAddressMode address_mode, VkFilter mip_filter);

	~Sampler();

	Sampler(Sampler &&other);

	Sampler &operator=(Sampler &&other);

	Sampler(const Sampler &) = delete;

	Sampler &operator=(const Sampler &other) = delete;

	const VkSampler &getSampler() const;

	operator const VkSampler &() const;

  private:
	VkSampler m_handle = VK_NULL_HANDLE;
};

using SamplerRefernece = std::reference_wrapper<VkSampler>;
}        // namespace Ilum