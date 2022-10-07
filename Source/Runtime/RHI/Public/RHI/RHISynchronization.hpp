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

	virtual void Wait(uint64_t timeout = std::numeric_limits<uint64_t>::max()) = 0;
	virtual void Reset()                = 0;

  protected:
	RHIDevice *p_device = nullptr;
};

class RHISemaphore
{
  public:
	RHISemaphore(RHIDevice *device);

	virtual ~RHISemaphore() = default;

	virtual void SetName(const std::string &name) = 0;

	static std::unique_ptr<RHISemaphore> Create(RHIDevice *device);

  protected:
	RHIDevice *p_device = nullptr;
};
}        // namespace Ilum