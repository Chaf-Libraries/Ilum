#pragma once

#include "RHI/RHIFrame.hpp"

#include <volk.h>

#include <vector>
#include <unordered_map>
#include <thread>

namespace Ilum::Vulkan
{
class Fence;
class Semaphore;
class Command;

class Frame : public RHIFrame
{
  public:
	Frame(RHIDevice *device);

	virtual ~Frame() override;

	virtual RHIFence *AllocateFence() override;

	virtual RHISemaphore *AllocateSemaphore() override;

	virtual RHICommand *AllocateCommand(RHIQueueFamily family) override;

	virtual void Reset() override;

  private:
	std::vector<std::unique_ptr<Fence>> m_fences;
	std::vector<std::unique_ptr<Semaphore>> m_semaphores;
	std::unordered_map<size_t, std::vector<std::unique_ptr<Command>>> m_commands;

	std::unordered_map<size_t, VkCommandPool> m_command_pools;

	uint32_t m_active_fence_index = 0;
	uint32_t m_active_semaphore_index = 0;
	std::unordered_map<size_t, uint32_t> m_active_cmd_index;
};
}        // namespace Ilum::Vulkan