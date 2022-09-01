#pragma once

#include "RHI/RHISampler.hpp"

namespace Ilum::CUDA
{
class Sampler : public RHISampler
{
  public:
	Sampler(RHIDevice *device, const SamplerDesc &desc);

	virtual ~Sampler() = default;

	void *GetHandle() const;
};
}        // namespace Ilum::CUDA