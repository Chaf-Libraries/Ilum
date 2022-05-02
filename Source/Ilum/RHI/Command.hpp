#pragma once

#include "Texture.hpp"

#include <volk.h>

#include <thread>
#include <vector>

namespace Ilum
{
class CommandBuffer;
class RHIDevice;

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
	CommandPool(RHIDevice *device, VkQueueFlagBits queue, ResetMode reset_mode = ResetMode::ResetPool, const std::thread::id &thread_id = std::this_thread::get_id());
	~CommandPool();

	CommandPool(const CommandPool &) = delete;
	CommandPool &operator=(const CommandPool &) = delete;
	CommandPool(CommandPool &&)                 = delete;
	CommandPool &operator=(CommandPool &&) = delete;

	operator const VkCommandPool &() const;

	size_t Hash() const;

	void Reset();

	ResetMode GetResetMode() const;

	CommandBuffer &RequestCommandBuffer(VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

  private:
	RHIDevice *p_device = nullptr;

	VkCommandPool   m_handle = VK_NULL_HANDLE;
	std::thread::id m_thread_id;
	VkQueueFlagBits m_queue;
	ResetMode       m_reset_mode;
	size_t          m_hash = 0;

	std::vector<std::unique_ptr<CommandBuffer>> m_primary_cmd_buffers;
	std::vector<std::unique_ptr<CommandBuffer>> m_secondary_cmd_buffers;

	uint32_t m_active_primary_count   = 0;
	uint32_t m_active_secondary_count = 0;
};

class CommandBuffer
{
  public:
	CommandBuffer(RHIDevice *device, CommandPool *pool, VkCommandBufferLevel level);
	~CommandBuffer();

	CommandBuffer(const CommandBuffer &) = delete;
	CommandBuffer &operator=(const CommandBuffer &) = delete;

	void Reset() const;

	void Begin(VkCommandBufferUsageFlagBits usage = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, VkCommandBufferInheritanceInfo *inheritanceInfo = nullptr);
	void End();

	void BeginRenderPass(VkRenderPass pass, const VkRect2D& area, VkFramebuffer framebuffer, const std::vector<VkClearValue>& clear_values);
	void EndRenderPass();

	void Transition(Texture *texture, const TextureState &src, const TextureState &dst, const VkImageSubresourceRange &range);

	operator const VkCommandBuffer &() const;

  private:
	RHIDevice   *p_device = nullptr;
	CommandPool *p_pool   = nullptr;

	VkCommandBuffer m_handle = VK_NULL_HANDLE;
};

}        // namespace Ilum