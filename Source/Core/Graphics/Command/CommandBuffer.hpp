#pragma once

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
class CommandPool;
class LogicalDevice;

class CommandBuffer
{
  public:
	enum class State
	{
		Initial,
		Recording,
		Executable,
		Invalid
	};

  public:
	CommandBuffer(const CommandPool &command_pool, VkQueueFlagBits queue_type = VK_QUEUE_GRAPHICS_BIT, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

	~CommandBuffer();

	void reset();

	bool begin(VkCommandBufferUsageFlagBits usage = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	void end();

	void submitIdle(uint32_t queue_index = 0);

	void submit(const VkSemaphore &wait_semaphore = VK_NULL_HANDLE, const VkSemaphore &signal_semaphore = VK_NULL_HANDLE, VkFence fence = VK_NULL_HANDLE, VkShaderStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, uint32_t queue_index = 0);

	void submit(const std::vector<VkSemaphore> &wait_semaphores = {}, const std::vector<VkSemaphore> &signal_semaphores = {}, VkFence fence = VK_NULL_HANDLE, const std::vector<VkShaderStageFlags> &wait_stages = {}, uint32_t queue_index = 0);

	operator const VkCommandBuffer &() const;

	const VkCommandBuffer &getCommandBuffer() const;

	const State &getState() const;

  private:
	const CommandPool &  m_command_pool;
	const LogicalDevice &m_logical_device;
	VkCommandBuffer      m_handle = VK_NULL_HANDLE;
	VkQueueFlagBits      m_queue_type;
	State                m_state = State::Initial;
};
}        // namespace Ilum