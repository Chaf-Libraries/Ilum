#pragma once

#include "Fwd.hpp"

namespace Ilum
{
ENUM(SamplerStateType, Enable){
    LinearClamp,
    LinearWarp,
    NearestClamp,
    NearestWarp,
};

struct SamplerDesc
{
	RHIFilter min_filter;
	RHIFilter mag_filter;

	RHIAddressMode address_mode_u;
	RHIAddressMode address_mode_v;
	RHIAddressMode address_mode_w;
	RHIMipmapMode  mipmap_mode;

	RHISamplerBorderColor border_color = RHISamplerBorderColor::Float_Transparent_Black;

	bool anisotropic = true;

	float mip_lod_bias = 0.f;
	float min_lod      = 0.f;
	float max_lod      = 100.f;

	template<typename Archive>
	void serialize(Archive& archive)
	{
		archive(min_filter, mag_filter, address_mode_u, address_mode_v, address_mode_w, mipmap_mode, border_color, anisotropic, mip_lod_bias, min_lod, max_lod);
	}

	static SamplerDesc LinearClamp();
	static SamplerDesc LinearWarp();
	static SamplerDesc NearestClamp();
	static SamplerDesc NearestWarp();
};

class RHISampler
{
  public:
	RHISampler(RHIDevice *device, const SamplerDesc &desc);

	virtual ~RHISampler() = default;

	static std::unique_ptr<RHISampler> Create(RHIDevice *device, const SamplerDesc &desc);

  protected:
	RHIDevice  *p_device = nullptr;
	SamplerDesc m_desc;
};
}        // namespace Ilum