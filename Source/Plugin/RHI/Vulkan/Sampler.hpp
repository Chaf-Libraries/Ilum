#pragma once

#include "Fwd.hpp"

namespace Ilum::Vulkan
{
class Sampler : public RHISampler
{
  public:
	Sampler(RHIDevice *device, const SamplerDesc &desc);
	virtual ~Sampler() override;

	VkSampler GetHandle() const;

  private:
	VkSampler m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum::Vulkan