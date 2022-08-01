#pragma once

#include <cstdint>

namespace Ilum
{
class RHIDevice;

class RHIFence
{
  public:
	RHIFence(RHIDevice *device);
	virtual ~RHIFence() = default;

	static std::unique_ptr<RHIFence> Create(RHIDevice *device);

	virtual void Wait(uint32_t timeout) = 0;
	virtual void Reset()                = 0;

  protected:
	RHIDevice *p_device = nullptr;
};

class RHISemaphore
{
  public:
	RHISemaphore(RHIDevice *device);
	virtual ~RHISemaphore() = default;

	static std::unique_ptr<RHISemaphore> Create(RHIDevice *device);

  protected:
	RHIDevice *p_device = nullptr;
};
}        // namespace Ilum