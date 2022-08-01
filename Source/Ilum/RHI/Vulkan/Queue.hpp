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

	virtual void Submit(const std::vector<RHICommond *> &cmds, const std::vector<RHISemaphore *> &signal_semaphores = {}, const std::vector<RHISemaphore *> &wait_semaphores = {}) override;

	virtual void Execute(RHIFence *fence = nullptr) override;

	VkQueue GetHandle() const;

  private:
	VkQueue m_handle = VK_NULL_HANDLE;
	std::vector<std::tuple<std::vector<RHICommond *>, std::vector<RHISemaphore *>, std::vector<RHISemaphore *>>> m_submits;
};
}        // namespace Ilum::Vulkan