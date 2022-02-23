#pragma once

#include "Graphics/Synchronization/QueueSystem.hpp"

namespace Ilum
{
class CommandBuffer;

class CommandPool
{
  public:
	enum class ResetMode
	{
		ResetPool,
		ResetIndividually,
		AlwaysAllocate,
	};

  public:
	CommandPool(QueueUsage queue, ResetMode reset_mode = ResetMode::ResetPool, const std::thread::id &thread_id = std::this_thread::get_id());
	~CommandPool();

	CommandPool(const CommandPool &) = delete;
	CommandPool &operator=(const CommandPool &) = delete;
	CommandPool(CommandPool &&)                 = delete;
	CommandPool &operator=(CommandPool &&) = delete;

	operator const VkCommandPool &() const;

	const VkCommandPool &  getCommandPool() const;
	const std::thread::id &getThreadID() const;
	const QueueUsage &     getUsage() const;
	size_t                 getHash() const;
	ResetMode              getResetMode() const;

	void reset();

	CommandBuffer &requestCommandBuffer(VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

  private:
	VkCommandPool   m_handle = VK_NULL_HANDLE;
	std::thread::id m_thread_id;
	QueueUsage      m_queue;
	ResetMode       m_reset_mode;
	size_t          m_hash = 0;

	std::vector<scope<CommandBuffer>> m_primary_cmd_buffers;
	std::vector<scope<CommandBuffer>> m_secondary_cmd_buffers;

	uint32_t m_active_primary_count   = 0;
	uint32_t m_active_secondary_count = 0;
};
}        // namespace Ilum::Graphics