#include "Sampler.hpp"

namespace Ilum::CUDA
{
Sampler::Sampler(RHIDevice *device, const SamplerDesc &desc):
    RHISampler(device, desc)
{
}

void *Sampler::GetHandle() const
{
	return nullptr;
}
}        // namespace Ilum::CUDA