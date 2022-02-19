#pragma once

#include "Graphics/Vulkan.hpp"

namespace Ilum::Graphics
{
class CommandBuffer;
class Device;

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
	CommandPool(const Device &device, QueueFamily queue, ResetMode reset_mode = ResetMode::ResetPool, const std::thread::id &thread_id = std::this_thread::get_id());
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
	const Device &  m_device;
	VkCommandPool   m_handle = VK_NULL_HANDLE;
	std::thread::id m_thread_id;
	QueueFamily     m_queue;
	ResetMode       m_reset_mode;
	size_t          m_hash = 0;

	std::vector<std::unique_ptr<CommandBuffer>> m_primary_cmd_buffers;
	std::vector<std::unique_ptr<CommandBuffer>> m_secondary_cmd_buffers;

	uint32_t m_active_primary_count   = 0;
	uint32_t m_active_secondary_count = 0;
};
}        // namespace Ilum::Graphics