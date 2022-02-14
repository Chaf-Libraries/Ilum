#pragma once

#include "../Vulkan.hpp"

namespace Ilum::Graphics
{
class CommandPool;

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
	CommandBuffer(const CommandPool &cmd_pool, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	~CommandBuffer();

	CommandBuffer(const CommandBuffer &) = delete;
	CommandBuffer &operator=(const CommandBuffer &) = delete;
	CommandBuffer(CommandBuffer &&)                 = delete;
	CommandBuffer &operator=(CommandBuffer &&) = delete;

	operator const VkCommandBuffer &() const;

	const VkCommandBuffer &GetHandle() const;
	const State &          GetState() const;
	VkCommandBufferLevel   GetLevel() const;

	// Command buffer recording
	void Reset();
	void Begin(VkCommandBufferUsageFlagBits usage = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, VkCommandBufferInheritanceInfo *inheritanceInfo = nullptr);
	void End();
	void SubmitIdle(uint32_t queue_index = 0);

	// Copy image and buffer

  private:
	const CommandPool &m_pool;

	VkCommandBuffer m_handle = VK_NULL_HANDLE;

	QueueFamily          m_queue;
	VkCommandBufferLevel m_level;
	State                m_state = State::Invalid;
};
}