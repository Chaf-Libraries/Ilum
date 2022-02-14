#pragma once

#include "../Vulkan.hpp"

namespace Ilum::Graphics
{
class Device;

class SemaphorePool
{
  public:
	SemaphorePool(const Device &device);
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
	const Device &m_device;

	std::vector<VkSemaphore> m_semaphores;
	std::vector<VkSemaphore> m_released_semaphores;

	uint32_t m_active_semaphore_count = 0;

	std::mutex m_request_mutex;
	std::mutex m_release_mutex;
};
}        // namespace Ilum::Graphics