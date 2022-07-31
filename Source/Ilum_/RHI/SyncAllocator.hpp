#pragma once

#include <volk.h>

#include <mutex>
#include <vector>

namespace Ilum
{
class RHIDevice;

class FenceAllocator
{
  public:
	FenceAllocator(RHIDevice *device);
	~FenceAllocator();

	FenceAllocator(const FenceAllocator &) = delete;
	FenceAllocator &operator=(const FenceAllocator &) = delete;
	FenceAllocator(FenceAllocator &&)                 = delete;
	FenceAllocator &operator=(FenceAllocator &&) = delete;

	VkFence &RequestFence();

	void Wait(uint32_t timeout = std::numeric_limits<uint32_t>::max()) const;
	void Reset();

  private:
	RHIDevice *p_device = nullptr;

	std::vector<VkFence> m_fences;

	uint32_t m_active_fence_count = 0;

	std::mutex m_mutex;
};

class SemaphoreAllocator
{
  public:
	SemaphoreAllocator(RHIDevice *device);
	~SemaphoreAllocator();

	SemaphoreAllocator(const SemaphoreAllocator &) = delete;
	SemaphoreAllocator &operator=(const SemaphoreAllocator &) = delete;
	SemaphoreAllocator(SemaphoreAllocator &&)                 = delete;
	SemaphoreAllocator &operator=(SemaphoreAllocator &&) = delete;

	// Request a semaphore without ownership
	VkSemaphore RequestSemaphore();
	// Allocate a semaphore with ownership, you should release manually or call ReleaseSemaphre() to collect
	VkSemaphore AllocateSemaphore();
	void        ReleaseAllocatedSemaphore(VkSemaphore semaphore);

	void Reset();

  private:
	RHIDevice *p_device;

	std::vector<VkSemaphore> m_semaphores;
	std::vector<VkSemaphore> m_released_semaphores;

	uint32_t m_active_semaphore_count = 0;

	std::mutex m_request_mutex;
	std::mutex m_release_mutex;
};
}        // namespace Ilum