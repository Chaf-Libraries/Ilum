#pragma once

#include "RHI/RHIFrame.hpp"

namespace Ilum::CUDA
{
class Command;

class Frame : public RHIFrame
{
  public:
	Frame(RHIDevice *device);

	~Frame() = default;

	RHIFence *AllocateFence();

	RHISemaphore *AllocateSemaphore();

	RHICommand *AllocateCommand(RHIQueueFamily family);

	void Reset();

  private:
	std::vector<std::unique_ptr<Command>> m_cmds;
	uint32_t                              m_current_cmd = 0;
};
}        // namespace Ilum::CUDA