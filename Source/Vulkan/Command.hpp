#pragma once

#include "Vulkan.hpp"

namespace Ilum::Vulkan
{
class CommandBuffer;

// Vulkan Command Pool
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
	CommandPool(QueueFamily queue, ResetMode reset_mode = ResetMode::ResetPool, const std::thread::id &thread_id = std::this_thread::get_id());

	~CommandPool();

	CommandPool(const CommandPool &) = delete;
	CommandPool &operator=(const CommandPool &) = delete;
	CommandPool(CommandPool &&)                 = delete;
	CommandPool &operator=(CommandPool &&) = delete;

	operator const VkCommandPool &() const;

	const VkCommandPool &  GetHandle() const;
	const std::thread::id &GetThreadID() const;
	const QueueFamily &    GetQueueFamily() const;
	size_t                 GetHash() const;
	ResetMode              GetResetMode() const;

	void Reset();

	CommandBuffer &RequestCommandBuffer(VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

  private:
	VkCommandPool   m_handle = VK_NULL_HANDLE;
	std::thread::id m_thread_id;
	QueueFamily     m_queue;
	ResetMode       m_reset_mode;
	size_t          m_hash = 0;

	std::vector<std::unique_ptr<CommandBuffer>> m_cmd_buffers;
	uint32_t                                    m_active_count = 0;
};

// Vulkan Command Buffer
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

	void Reset();
	void Begin();
	void End();
	void SubmitIdle();

  private:
	const CommandPool &m_pool;

	VkCommandBuffer m_handle = VK_NULL_HANDLE;

	QueueFamily          m_queue;
	VkCommandBufferLevel m_level;
	State                m_state = State::Invalid;
};
}        // namespace Ilum::Vulkan