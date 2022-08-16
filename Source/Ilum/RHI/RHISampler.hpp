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
	RHISamplerBorderColor border_color = RHISamplerBorderColor::Float_Transparent_Black;

	bool anisotropic = false;

	float mip_lod_bias = 0.f;
	float min_lod = 0.f;
	float max_lod = 100.f;
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