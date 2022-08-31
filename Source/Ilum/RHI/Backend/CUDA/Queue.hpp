#pragma once

#include "RHI/RHIQueue.hpp"

namespace Ilum::CUDA
{
class Queue : public RHIQueue
{
  public:
	Queue(RHIDevice *device, RHIQueueFamily family, uint32_t queue_index = 0);

	~Queue() = default;

	virtual void Wait() override;

	virtual void Submit(const std::vector<RHICommand *> &cmds, const std::vector<RHISemaphore *> &signal_semaphores = {}, const std::vector<RHISemaphore *> &wait_semaphores = {}) override;

	virtual void Execute(RHIFence *fence = nullptr) override;

  private:
	std::vector<RHICommand *> m_cmds;
};
}        // namespace Ilum::CUDA