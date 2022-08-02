#pragma once

#include "RHIDefinitions.hpp"

#include <vector>

namespace Ilum
{
class RHIDevice;
class RHICommand;
class RHISemaphore;
class RHIFence;

class RHIQueue
{
  public:
	RHIQueue(RHIDevice *device, RHIQueueFamily family, uint32_t queue_index = 0);
	virtual ~RHIQueue() = default;

	static std::unique_ptr<RHIQueue> Create(RHIDevice *device, RHIQueueFamily family, uint32_t queue_index = 0);

	virtual void Submit(const std::vector<RHICommand *> &cmds, const std::vector<RHISemaphore *> &signal_semaphores = {}, const std::vector<RHISemaphore *> &wait_semaphores = {}) = 0;

	virtual void Execute(RHIFence *fence = nullptr) = 0;

  protected:
	RHIDevice     *p_device = nullptr;
	RHIQueueFamily m_family;
	uint32_t       m_queue_index = 0;
};
}        // namespace Ilum