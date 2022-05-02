#pragma once

#include <volk.h>

namespace Ilum
{
class RHIDevice;

struct SamplerDesc
{
	VkFilter min_filter;
	VkFilter mag_filter;

	VkSamplerAddressMode address_mode;
	VkSamplerMipmapMode  mipmap_mode;

	float mip_lod_bias = 0.f;
	float min_lod      = 0.f;
	float max_lod      = 0.f;
};

class Sampler
{
  public:
	Sampler(RHIDevice *device, const SamplerDesc &desc);
	~Sampler();

	operator VkSampler() const;

  private:
	RHIDevice *p_device = nullptr;
	SamplerDesc m_desc;
	VkSampler   m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum