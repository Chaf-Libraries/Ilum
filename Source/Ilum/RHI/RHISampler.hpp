#pragma once

#include "RHIDefinitions.hpp"

namespace Ilum
{
class RHIDevice;

struct SamplerDesc
{
	RHIFilter min_filter;
	RHIFilter mag_filter;

	RHIAddressMode address_mode_u;
	RHIAddressMode address_mode_v;
	RHIAddressMode address_mode_w;
	RHIMipmapMode  mipmap_mode;
	RHISamplerBorderColor border_color;

	bool anisotropic;

	float mip_lod_bias;
	float min_lod;
	float max_lod;
};

class RHISampler
{
  public:
	RHISampler(RHIDevice *device, const SamplerDesc &desc);
	virtual ~RHISampler() = default;

	static std::unique_ptr<RHISampler> Create(RHIDevice *device, const SamplerDesc &desc);

  protected:
	RHIDevice *p_device = nullptr;
	SamplerDesc m_desc;
};
}        // namespace Ilum