#pragma once

#include "RHIDefinitions.hpp"

#include <vector>

namespace Ilum
{
class RHIDevice;
class RHICommand;
class RHISemaphore;
class RHIFence;

struct [[serialization(false), reflection(false)]] SubmitInfo
{
	RHIQueueFamily queue_family;
	bool           is_cuda;

	std::vector<RHICommand *>   cmd_buffers;
	std::vector<RHISemaphore *> wait_semaphores;
	std::vector<RHISemaphore *> signal_semaphores;
};

class RHIQueue
{
  public:
	RHIQueue(RHIDevice *device);

	virtual ~RHIQueue() = default;

	static std::unique_ptr<RHIQueue> Create(RHIDevice *device);

	virtual void Execute(RHIQueueFamily family, const std::vector<SubmitInfo> &submit_infos, RHIFence* fence = nullptr) = 0;

	// Immediate execution
	virtual void Execute(RHICommand *cmd_buffer) = 0;

	virtual void Wait() = 0;
};
}        // namespace Ilum