#pragma once

#include "RHI/RHIQueue.hpp"

namespace Ilum::CUDA
{
class Queue : public RHIQueue
{
  public:
	Queue(RHIDevice *device);

	~Queue() = default;

	virtual void Execute(RHIQueueFamily family, const std::vector<SubmitInfo> &submit_infos, RHIFence *fence) override;

	// Immediate execution
	virtual void Execute(RHICommand *cmd_buffer) override;

	virtual void Wait() override;
};
}        // namespace Ilum::CUDA