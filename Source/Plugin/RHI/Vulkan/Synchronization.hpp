#pragma once

#include "Fwd.hpp"

namespace Ilum::Vulkan
{
class Fence : public RHIFence
{
  public:
	Fence(RHIDevice *device);
	virtual ~Fence() override;

	virtual void Wait(uint64_t timeout = std::numeric_limits<uint64_t>::max()) override;
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

	virtual void SetName(const std::string &name) override;

	VkSemaphore GetHandle() const;

  private:
	VkSemaphore m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum::Vulkan