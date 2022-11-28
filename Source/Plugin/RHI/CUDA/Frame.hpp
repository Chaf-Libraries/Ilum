#pragma once

#include "Fwd.hpp"

namespace Ilum::CUDA
{
class Command;

class Frame : public RHIFrame
{
  public:
	Frame(RHIDevice *device);

	~Frame() = default;

	virtual RHIFence *AllocateFence() override;

	virtual RHISemaphore *AllocateSemaphore() override;

	virtual RHICommand *AllocateCommand(RHIQueueFamily family) override;

	virtual RHIDescriptor* AllocateDescriptor(const ShaderMeta& meta) override
	{
		return nullptr;
	}

	void Reset();

  private:
	std::vector<std::unique_ptr<Command>> m_cmds;
	uint32_t                              m_current_cmd = 0;
};
}        // namespace Ilum::CUDA