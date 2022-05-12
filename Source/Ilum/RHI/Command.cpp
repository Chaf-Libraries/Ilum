#include "Command.hpp"
#include "DescriptorState.hpp"
#include "Device.hpp"
#include "FrameBuffer.hpp"
#include "PipelineState.hpp"

#include "Command.hpp"
#include <Core/Hash.hpp>
#include <Core/Macro.hpp>

namespace Ilum
{
CommandPool::CommandPool(RHIDevice *device, VkQueueFlagBits queue, ResetMode reset_mode, const std::thread::id &thread_id) :
    p_device(device),
    m_queue(queue),
    m_thread_id(thread_id),
    m_reset_mode(reset_mode)
{
	m_hash = 0;
	HashCombine(m_hash, m_queue);
	HashCombine(m_hash, reset_mode);
	HashCombine(m_hash, thread_id);

	VkCommandPoolCreateInfo create_info = {};
	create_info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	create_info.flags                   = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
	switch (m_queue)
	{
		case VK_QUEUE_GRAPHICS_BIT:
			create_info.queueFamilyIndex = p_device->GetGraphicsFamily();
			break;
		case VK_QUEUE_COMPUTE_BIT:
			create_info.queueFamilyIndex = p_device->GetComputeFamily();
			break;
		case VK_QUEUE_TRANSFER_BIT:
			create_info.queueFamilyIndex = p_device->GetTransferFamily();
			break;
		default:
			break;
	}

	switch (reset_mode)
	{
		case ResetMode::ResetIndividually:
		case ResetMode::AlwaysAllocate:
			create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
			break;
		case ResetMode::ResetPool:
		default:
			create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
			break;
	}

	vkCreateCommandPool(p_device->GetDevice(), &create_info, nullptr, &m_handle);
}

CommandPool::~CommandPool()
{
	m_primary_cmd_buffers.clear();
	m_secondary_cmd_buffers.clear();
	vkDestroyCommandPool(p_device->GetDevice(), m_handle, nullptr);
}

CommandPool::operator const VkCommandPool &() const
{
	return m_handle;
}

size_t CommandPool::Hash() const
{
	return m_hash;
}

void CommandPool::Reset()
{
	switch (m_reset_mode)
	{
		case ResetMode::ResetIndividually:
			for (auto &cmd_buffer : m_primary_cmd_buffers)
			{
				cmd_buffer->Reset();
			}
			for (auto &cmd_buffer : m_secondary_cmd_buffers)
			{
				cmd_buffer->Reset();
			}
			break;
		case ResetMode::ResetPool:
			vkResetCommandPool(p_device->GetDevice(), m_handle, 0);
			for (auto &cmd_buffer : m_primary_cmd_buffers)
			{
				cmd_buffer->Reset();
			}
			for (auto &cmd_buffer : m_secondary_cmd_buffers)
			{
				cmd_buffer->Reset();
			}
			break;
		case ResetMode::AlwaysAllocate:
			m_primary_cmd_buffers.clear();
			m_secondary_cmd_buffers.clear();
			break;
		default:
			LOG_FATAL("Unknown reset mode for command pools");
	}

	m_active_primary_count   = 0;
	m_active_secondary_count = 0;
}

CommandPool::ResetMode CommandPool::GetResetMode() const
{
	return m_reset_mode;
}

CommandBuffer &CommandPool::RequestCommandBuffer(VkCommandBufferLevel level)
{
	if (level == VK_COMMAND_BUFFER_LEVEL_PRIMARY)
	{
		if (m_active_primary_count < m_primary_cmd_buffers.size())
		{
			return *m_primary_cmd_buffers.at(m_active_primary_count++);
		}

		m_primary_cmd_buffers.emplace_back(std::make_unique<CommandBuffer>(p_device, this, level));

		m_active_primary_count++;

		return *m_primary_cmd_buffers.back();
	}
	else
	{
		if (m_active_secondary_count < m_secondary_cmd_buffers.size())
		{
			return *m_secondary_cmd_buffers.at(m_active_secondary_count++);
		}

		m_secondary_cmd_buffers.emplace_back(std::make_unique<CommandBuffer>(p_device, this, level));

		m_active_secondary_count++;

		return *m_secondary_cmd_buffers.back();
	}
}

CommandBuffer::CommandBuffer(RHIDevice *device, CommandPool *pool, VkCommandBufferLevel level) :
    p_device(device), p_pool(pool)
{
	VkCommandBufferAllocateInfo command_buffer_allocate_info = {};
	command_buffer_allocate_info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	command_buffer_allocate_info.commandPool                 = *pool;
	command_buffer_allocate_info.level                       = level;
	command_buffer_allocate_info.commandBufferCount          = 1;
	vkAllocateCommandBuffers(p_device->GetDevice(), &command_buffer_allocate_info, &m_handle);
}

CommandBuffer::~CommandBuffer()
{
	if (m_handle)
	{
		vkFreeCommandBuffers(p_device->GetDevice(), *p_pool, 1, &m_handle);
	}
}

void CommandBuffer::Reset() const
{
	if (p_pool->GetResetMode() == CommandPool::ResetMode::ResetIndividually)
	{
		vkResetCommandBuffer(m_handle, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
	}
}

void CommandBuffer::Begin(VkCommandBufferUsageFlagBits usage, VkCommandBufferInheritanceInfo *inheritanceInfo)
{
	VkCommandBufferBeginInfo command_buffer_begin_info = {};
	command_buffer_begin_info.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	command_buffer_begin_info.flags                    = usage;
	command_buffer_begin_info.pInheritanceInfo         = inheritanceInfo;

	vkBeginCommandBuffer(m_handle, &command_buffer_begin_info);
}

void CommandBuffer::End()
{
	vkEndCommandBuffer(m_handle);
}

void CommandBuffer::BeginRenderPass(FrameBuffer &frame_buffer)
{
	m_current_fb = &frame_buffer;

	VkRect2D rect      = {};
	rect.extent.width  = frame_buffer.GetWidth();
	rect.extent.height = frame_buffer.GetHeight();

	VkRenderPassBeginInfo begin_info = {};
	begin_info.sType                 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	begin_info.renderPass            = p_device->AllocateRenderPass(frame_buffer);
	begin_info.renderArea            = rect;
	begin_info.framebuffer           = p_device->AllocateFrameBuffer(frame_buffer);
	begin_info.clearValueCount       = static_cast<uint32_t>(frame_buffer.GetClearValue().size());
	begin_info.pClearValues          = frame_buffer.GetClearValue().data();

	vkCmdBeginRenderPass(m_handle, &begin_info, VK_SUBPASS_CONTENTS_INLINE);
}

void CommandBuffer::EndRenderPass()
{
	vkCmdEndRenderPass(m_handle);
}

void CommandBuffer::Bind(const PipelineState &pso)
{
	m_current_pso = &pso;
	vkCmdBindPipeline(m_handle, pso.GetBindPoint(), p_device->AllocatePipeline(*m_current_pso, m_current_fb == nullptr ? VK_NULL_HANDLE : p_device->AllocateRenderPass(*m_current_fb)));
}

void CommandBuffer::Bind(DescriptorState &descriptor_state)
{
	ASSERT(m_current_pso);
	descriptor_state.Write();
	for (auto &[set, descriptor_set] : descriptor_state.m_descriptor_sets)
	{
		vkCmdBindDescriptorSets(m_handle, m_current_pso->GetBindPoint(), p_device->AllocatePipelineLayout(*m_current_pso), set, 1, &descriptor_set, 0, nullptr);
	}
}

DescriptorState &CommandBuffer::GetDescriptorState() const
{
	ASSERT(m_current_pso);
	return p_device->AllocateDescriptorState(*m_current_pso);
}

void CommandBuffer::Transition(Texture *texture, const TextureState &src, const TextureState &dst, const VkImageSubresourceRange &range)
{
	VkImageMemoryBarrier barrier = {};
	barrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.srcAccessMask        = src.access_mask;
	barrier.dstAccessMask        = dst.access_mask;
	barrier.oldLayout            = src.layout;
	barrier.newLayout            = dst.layout;
	barrier.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                = *texture;
	barrier.subresourceRange     = range;

	vkCmdPipelineBarrier(*this, src.stage, dst.stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void CommandBuffer::Transition(Buffer *buffer, const BufferState &src, const BufferState &dst)
{
	VkBufferMemoryBarrier barrier = {};
	barrier.sType                 = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.srcAccessMask         = src.access_mask;
	barrier.dstAccessMask         = dst.access_mask;
	barrier.srcQueueFamilyIndex   = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex   = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer                = *buffer;
	barrier.offset                = 0;
	barrier.size                  = buffer->GetSize();

	vkCmdPipelineBarrier(*this, src.stage, dst.stage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void CommandBuffer::Transition(const std::vector<BufferTransition> &buffer_transitions, const std::vector<TextureTransition> &texture_transitions)
{
	std::vector<VkBufferMemoryBarrier> buffer_barriers(buffer_transitions.size());
	std::vector<VkImageMemoryBarrier>  image_barriers(texture_transitions.size());

	VkPipelineStageFlags src_stage = 0;
	VkPipelineStageFlags dst_stage = 0;

	for (uint32_t i = 0; i < buffer_barriers.size(); i++)
	{
		buffer_barriers[i].sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		buffer_barriers[i].srcAccessMask       = buffer_transitions[i].src.access_mask;
		buffer_barriers[i].dstAccessMask       = buffer_transitions[i].dst.access_mask;
		buffer_barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		buffer_barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		buffer_barriers[i].buffer              = *buffer_transitions[i].buffer;
		buffer_barriers[i].offset              = 0;
		buffer_barriers[i].size                = buffer_transitions[i].buffer->GetSize();
		src_stage |= buffer_transitions[i].src.stage;
		dst_stage |= buffer_transitions[i].dst.stage;
	}

	for (uint32_t i = 0; i < image_barriers.size(); i++)
	{
		image_barriers[i].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		image_barriers[i].srcAccessMask       = texture_transitions[i].src.access_mask;
		image_barriers[i].dstAccessMask       = texture_transitions[i].dst.access_mask;
		image_barriers[i].oldLayout           = texture_transitions[i].src.layout;
		image_barriers[i].newLayout           = texture_transitions[i].dst.layout;
		image_barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		image_barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		image_barriers[i].image               = *texture_transitions[i].texture;
		image_barriers[i].subresourceRange    = texture_transitions[i].range;
		src_stage |= texture_transitions[i].src.stage;
		dst_stage |= texture_transitions[i].dst.stage;
	}

	vkCmdPipelineBarrier(*this, src_stage, dst_stage, 0, 0, nullptr, static_cast<uint32_t>(buffer_barriers.size()), buffer_barriers.data(), static_cast<uint32_t>(image_barriers.size()), image_barriers.data());
}

void CommandBuffer::Dispatch(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z)
{
	vkCmdDispatch(m_handle, group_count_x, group_count_y, group_count_z);
}

void CommandBuffer::Draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance)
{
	vkCmdDraw(m_handle, vertex_count, instance_count, first_vertex, first_instance);
}

void CommandBuffer::DrawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, uint32_t vertex_offset, uint32_t first_instance)
{
	vkCmdDrawIndexed(m_handle, index_count, instance_count, first_index, vertex_offset, first_instance);
}

void CommandBuffer::SetViewport(float width, float height, float x, float y, float min_depth, float max_depth)
{
	VkViewport viewport = {x, y, width, height, min_depth, max_depth};
	vkCmdSetViewport(m_handle, 0, 1, &viewport);
}

void CommandBuffer::SetScissor(uint32_t width, uint32_t height, int32_t x, int32_t y)
{
	VkRect2D rect = {x, y, width, height};
	vkCmdSetScissor(m_handle, 0, 1, &rect);
}

void CommandBuffer::GenerateMipmap(Texture *texture, const TextureState &initial_state, VkFilter filter)
{
	VkImageSubresourceRange  src_range  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, texture->GetMipLevels(), 0, texture->GetLayerCount()};
	VkImageSubresourceRange  dst_range  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, texture->GetMipLevels(), 0, texture->GetLayerCount()};
	VkImageSubresourceLayers src_layers = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	VkImageSubresourceLayers dst_layers = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	VkImageUsageFlagBits     src_usage  = VK_IMAGE_USAGE_TRANSFER_DST_BIT;

	uint32_t src_width  = texture->GetWidth();
	uint32_t src_height = texture->GetHeight();
	uint32_t dst_width  = texture->GetWidth();
	uint32_t dst_height = texture->GetHeight();

	for (uint32_t i = 1; i < texture->GetMipLevels(); i++)
	{
		src_width  = dst_width;
		src_height = dst_height;

		dst_width  = std::max(src_width / 2, 1u);
		dst_height = std::max(src_height / 2, 1u);

		src_layers.mipLevel    = i - 1;
		src_range.baseMipLevel = i - 1;
		src_range.levelCount   = 1;

		dst_layers.mipLevel    = i;
		dst_range.baseMipLevel = i;
		dst_range.levelCount   = 1;

		std::vector<TextureTransition> transitions(2);
		transitions[0].texture = texture;
		transitions[0].src     = TextureState(src_usage);
		transitions[0].dst     = TextureState(VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
		transitions[0].range   = src_range;
		transitions[1].texture = texture;
		transitions[1].src     = TextureState();
		transitions[1].dst     = TextureState(VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		transitions[1].range   = dst_range;
		Transition({}, transitions);

		src_usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;

		VkImageBlit blit_info    = {};
		blit_info.srcOffsets[0]  = {0, 0, 0};
		blit_info.srcOffsets[1]  = {static_cast<int32_t>(src_width), static_cast<int32_t>(src_height), 1};
		blit_info.dstOffsets[0]  = {0, 0, 0};
		blit_info.dstOffsets[1]  = {static_cast<int32_t>(dst_width), static_cast<int32_t>(dst_height), 1};
		blit_info.srcSubresource = src_layers;
		blit_info.dstSubresource = dst_layers;

		vkCmdBlitImage(*this, *texture, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *texture, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit_info, filter);
	}

	VkImageSubresourceRange mip_level_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, texture->GetMipLevels(), 0, texture->GetLayerCount()};
	mip_level_range.levelCount              = mip_level_range.levelCount - 1;
	VkImageMemoryBarrier barrier            = {};
	barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.srcAccessMask                   = VK_ACCESS_TRANSFER_READ_BIT;
	barrier.dstAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.oldLayout                       = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	barrier.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                           = *texture;
	barrier.subresourceRange                = mip_level_range;

	vkCmdPipelineBarrier(*this, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void CommandBuffer::CopyBufferToImage(const BufferCopyInfo &buffer, const TextureCopyInfo &texture)
{
	VkBufferImageCopy copy_info = {};
	copy_info.bufferOffset      = buffer.offset;
	copy_info.bufferImageHeight = 0;
	copy_info.bufferRowLength   = 0;
	copy_info.imageSubresource  = texture.subresource;
	copy_info.imageOffset       = {0, 0, 0};
	copy_info.imageExtent       = {texture.texture->GetMipWidth(texture.subresource.mipLevel), texture.texture->GetMipHeight(texture.subresource.mipLevel), 1};

	vkCmdCopyBufferToImage(*this, *buffer.buffer, *texture.texture, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_info);
}

void CommandBuffer::CopyBuffer(const BufferCopyInfo &src, const BufferCopyInfo &dst, size_t size)
{
	VkBufferCopy copy_info = {};
	copy_info.size         = size;
	copy_info.srcOffset    = src.offset;
	copy_info.dstOffset    = dst.offset;

	vkCmdCopyBuffer(m_handle, *src.buffer, *dst.buffer, 1, &copy_info);
}

void CommandBuffer::BindVertexBuffer(Buffer *vertex_buffer)
{
	VkBuffer buffer_handle = *vertex_buffer;
	VkDeviceSize offsets[1]    = {0};
	vkCmdBindVertexBuffers(m_handle, 0, 1, &buffer_handle, offsets);
}

void CommandBuffer::BindIndexBuffer(Buffer *index_buffer)
{
	vkCmdBindIndexBuffer(m_handle, *index_buffer, 0, VK_INDEX_TYPE_UINT32);
}

void CommandBuffer::PushConstants(VkShaderStageFlags stage, void *data, uint32_t size, uint32_t offset)
{
	vkCmdPushConstants(m_handle, p_device->AllocatePipelineLayout(*m_current_pso), stage, offset, size, data);
}

void CommandBuffer::BeginMarker(const std::string &name, const glm::vec4 color)
{
	VkDebugUtilsLabelEXT label = {};
	label.sType                = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
	label.pLabelName           = name.c_str();
	label.color[0]             = color.r;
	label.color[1]             = color.g;
	label.color[2]             = color.b;
	label.color[3]             = color.a;
	label.pNext      = nullptr;
	vkCmdBeginDebugUtilsLabelEXT(m_handle, &label);
}

void CommandBuffer::EndMarker()
{
	vkCmdEndDebugUtilsLabelEXT(m_handle);
}

CommandBuffer::operator const VkCommandBuffer &() const
{
	return m_handle;
}
}        // namespace Ilum