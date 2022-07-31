#pragma once

#include <volk.h>

namespace Ilum
{
class RHIDevice;

struct SamplerDesc
{
	VkFilter min_filter = VK_FILTER_LINEAR;
	VkFilter mag_filter = VK_FILTER_LINEAR;

	VkSamplerAddressMode address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	VkSamplerMipmapMode  mipmap_mode  = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	bool anisotropic = false;

	float mip_lod_bias = 0.f;
	float min_lod      = 0.f;
	float max_lod      = std::numeric_limits<float>::max();

	size_t Hash() const
	{
		size_t hash = 0;
		HashCombine(hash, min_filter);
		HashCombine(hash, mag_filter);
		HashCombine(hash, address_mode);
		HashCombine(hash, mipmap_mode);
		HashCombine(hash, anisotropic);
		HashCombine(hash, mip_lod_bias);
		HashCombine(hash, min_lod);
		HashCombine(hash, max_lod);
		return hash;
	}
};

class Sampler
{
  public:
	Sampler(RHIDevice *device, const SamplerDesc &desc);
	~Sampler();

	operator VkSampler() const;

  private:
	RHIDevice  *p_device = nullptr;
	SamplerDesc m_desc;
	VkSampler   m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum