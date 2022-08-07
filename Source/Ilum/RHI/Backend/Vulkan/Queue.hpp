#pragma once

#include "RHI/RHIQueue.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
class Queue : public RHIQueue
{
  public:
	Queue(RHIDevice *device, RHIQueueFamily family, uint32_t queue_index = 0);

	virtual ~Queue() = default;

	virtual void Wait() override;

	virtual void Submit(const std::vector<RHICommand *> &cmds, const std::vector<RHISemaphore *> &signal_semaphores = {}, const std::vector<RHISemaphore *> &wait_semaphores = {}) override;

	virtual void Execute(RHIFence *fence = nullptr) override;

	VkQueue GetHandle() const;

  private:
	VkQueue m_handle = VK_NULL_HANDLE;

	std::vector<VkCommandBuffer> m_cmds;
	std::vector<VkSemaphore>     m_wait_semaphores;
	std::vector<VkSemaphore>     m_signal_semaphores;
};
}        // namespace Ilum::Vulkan