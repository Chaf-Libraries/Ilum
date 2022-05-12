#pragma once

#include "Buffer.hpp"
#include "Texture.hpp"

#include <volk.h>

#include <thread>
#include <vector>

namespace Ilum
{
class CommandBuffer;
class RHIDevice;
class DescriptorState;
class PipelineState;
class FrameBuffer;

struct BufferTransition
{
	Buffer     *buffer = nullptr;
	BufferState src    = {};
	BufferState dst    = {};
};

struct TextureTransition
{
	Texture                *texture = nullptr;
	TextureState            src     = {};
	TextureState            dst     = {};
	VkImageSubresourceRange range   = {};
};

struct BufferCopyInfo
{
	Buffer      *buffer = nullptr;
	VkDeviceSize offset = 0;
};

struct TextureCopyInfo
{
	Texture                 *texture     = nullptr;
	VkImageSubresourceLayers subresource = {};
};

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

	void BeginRenderPass(FrameBuffer &frame_buffer);
	void EndRenderPass();

	void Bind(const PipelineState &pso);
	void Bind(DescriptorState &descriptor_state);

	DescriptorState &GetDescriptorState() const;

	void Transition(Texture *texture, const TextureState &src, const TextureState &dst, const VkImageSubresourceRange &range);
	void Transition(Buffer *buffer, const BufferState &src, const BufferState &dst);
	void Transition(const std::vector<BufferTransition> &buffer_transitions, const std::vector<TextureTransition> &texture_transitions);

	void Dispatch(uint32_t group_count_x = 1, uint32_t group_count_y = 1, uint32_t group_count_z = 1);
	void Draw(uint32_t vertex_count, uint32_t instance_count = 1, uint32_t first_vertex = 0, uint32_t first_instance = 0);
	void DrawIndexed(uint32_t index_count, uint32_t instance_count = 1, uint32_t first_index = 0, uint32_t vertex_offset = 0, uint32_t first_instance = 0);

	void SetViewport(float width, float height, float x = 0.f, float y = 0.f, float min_depth = 0.f, float max_depth = 1.f);
	void SetScissor(uint32_t width, uint32_t height, int32_t x = 0, int32_t y = 0);

	void GenerateMipmap(Texture *texture, const TextureState &initial_state, VkFilter filter);

	void CopyBufferToImage(const BufferCopyInfo &buffer, const TextureCopyInfo &texture);
	void CopyBuffer(const BufferCopyInfo &src, const BufferCopyInfo &dst, size_t size);

	void BindVertexBuffer(Buffer *vertex_buffer);
	void BindIndexBuffer(Buffer *index_buffer);

	void PushConstants(VkShaderStageFlags stage, void *data, uint32_t size, uint32_t offset);

	void BeginMarker(const std::string &name, const glm::vec4 color);
	void EndMarker();

	operator const VkCommandBuffer &() const;

  private:
	RHIDevice   *p_device = nullptr;
	CommandPool *p_pool   = nullptr;

	VkCommandBuffer m_handle = VK_NULL_HANDLE;

	const PipelineState *m_current_pso = nullptr;
	FrameBuffer         *m_current_fb  = nullptr;
};

}        // namespace Ilum