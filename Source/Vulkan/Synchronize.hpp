#pragma once

#include "Vulkan.hpp"

namespace Ilum::Vulkan
{
class FencePool
{
  public:
	FencePool() = default;
	~FencePool();

	FencePool(const FencePool &) = delete;
	FencePool &operator=(const FencePool &) = delete;
	FencePool(FencePool &&)                 = delete;
	FencePool &operator=(FencePool &&) = delete;

	VkFence &RequestFence();

	void Wait(uint32_t timeout = std::numeric_limits<uint32_t>::max()) const;

	void Reset();

  private:
	std::vector<VkFence> m_fences;

	uint32_t m_active_fence_count = 0;

	std::mutex m_mutex;
};

class SemaphorePool
{
  public:
	SemaphorePool() = default;
	~SemaphorePool();

	SemaphorePool(const SemaphorePool &) = delete;
	SemaphorePool &operator=(const SemaphorePool &) = delete;
	SemaphorePool(SemaphorePool &&)                 = delete;
	SemaphorePool &operator=(SemaphorePool &&) = delete;

	// Request a semaphore without ownership
	VkSemaphore RequestSemaphore();
	// Allocate a semaphore with ownership, you should release manually or call ReleaseSemaphre() to collect
	VkSemaphore AllocateSemaphore();
	void        ReleaseAllocatedSemaphore(VkSemaphore semaphore);

	void Reset();

  private:
	std::vector<VkSemaphore> m_semaphores;
	std::vector<VkSemaphore> m_released_semaphores;

	uint32_t m_active_semaphore_count = 0;

	std::mutex m_request_mutex;
	std::mutex m_release_mutex;
};

}        // namespace Ilum::Vulkan