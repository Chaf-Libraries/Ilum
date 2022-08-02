#pragma once

#include "RHI/RHISynchronization.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
class Fence : public RHIFence
{
  public:
	Fence(RHIDevice *device);
	virtual ~Fence() override;

	virtual void Wait(uint64_t timeout) override;
	virtual void Reset() override;

	VkFence GetHandle() const;

  private:
	VkFence m_handle = VK_NULL_HANDLE;
};

class Semaphore : public RHISemaphore
{
  public:
	Semaphore(RHIDevice *device);
	virtual ~Semaphore() override;

	VkSemaphore GetHandle() const;

  private:
	VkSemaphore m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum::Vulkan