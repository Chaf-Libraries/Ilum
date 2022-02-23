#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
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
	VkSemaphore requestSemaphore();
	// Allocate a semaphore with ownership, you should release manually or call ReleaseSemaphre() to collect
	VkSemaphore allocateSemaphore();
	void        releaseAllocatedSemaphore(VkSemaphore semaphore);

	void reset();

  private:
	std::vector<VkSemaphore> m_semaphores;
	std::vector<VkSemaphore> m_released_semaphores;

	uint32_t m_active_semaphore_count = 0;

	std::mutex m_request_mutex;
	std::mutex m_release_mutex;
};
}        // namespace Ilum::Graphics