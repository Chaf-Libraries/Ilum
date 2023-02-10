#pragma once

#include "Fwd.hpp"

namespace Ilum
{
struct ShaderMeta;

class RHIFrame
{
  public:
	RHIFrame(RHIDevice *device);

	virtual ~RHIFrame() = default;

	static std::unique_ptr<RHIFrame> Create(RHIDevice *device);

	virtual RHIFence *AllocateFence() = 0;

	virtual RHISemaphore *AllocateSemaphore() = 0;

	virtual RHICommand *AllocateCommand(RHIQueueFamily family) = 0;

	virtual RHIDescriptor *AllocateDescriptor(const ShaderMeta &meta) = 0;

	virtual void Reset() = 0;

  protected:
	RHIDevice *p_device = nullptr;
};
}        // namespace Ilum