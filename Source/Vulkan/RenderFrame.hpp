#pragma once

#include "Command.hpp"
#include "Vulkan.hpp"

namespace Ilum::Vulkan
{
class RenderFrame
{
  public:
	RenderFrame();

	~RenderFrame();

	void Reset();

	CommandBuffer &RequestCommandBuffer(VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, QueueFamily queue = QueueFamily::Graphics, CommandPool::ResetMode reset_mode = CommandPool::ResetMode::ResetPool);

  private:
	CommandPool &RequestCommandPool(QueueFamily queue, CommandPool::ResetMode reset_mode = CommandPool::ResetMode::ResetPool);

  private:
	std::unordered_map<size_t, std::unique_ptr<CommandPool>> m_command_pools;
};
}        // namespace Ilum::Vulkan