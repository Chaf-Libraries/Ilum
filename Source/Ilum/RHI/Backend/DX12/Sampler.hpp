#pragma once

#include "RHI/RHISampler.hpp"

#include <d3d12.h>

namespace Ilum::DX12
{
class Sampler : public RHISampler
{
  public:
	Sampler(RHIDevice *device, const SamplerDesc &desc);

	virtual ~Sampler() override = default;

  private:
	D3D12_STATIC_SAMPLER_DESC m_handle;
};
}        // namespace Ilum::DX12