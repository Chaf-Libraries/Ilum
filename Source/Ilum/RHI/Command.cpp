#include "Command.hpp"
#include "DescriptorState.hpp"
#include "PipelineAllocator.hpp"
#include "Device.hpp"

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

	vkCreateCommandPool(p_device->m_device, &create_info, nullptr, &m_handle);
}

CommandPool::~CommandPool()
{
	m_primary_cmd_buffers.clear();
	m_secondary_cmd_buffers.clear();
	vkDestroyCommandPool(p_device->m_device, m_handle, nullptr);
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
			vkResetCommandPool(p_device->m_device, m_handle, 0);
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
	vkAllocateCommandBuffers(p_device->m_device, &command_buffer_allocate_info, &m_handle);
}

CommandBuffer::~CommandBuffer()
{
	if (m_handle)
	{
		vkFreeCommandBuffers(p_device->m_device, *p_pool, 1, &m_handle);
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
	VkRect2D rect = {};
	rect.extent.width = frame_buffer.m_width;
	rect.extent.height = frame_buffer.m_height;

	VkRenderPassBeginInfo begin_info = {};
	begin_info.sType                 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	begin_info.renderPass            = p_device->m_pipeline_allocator->CreateRenderPass(frame_buffer);
	begin_info.renderArea            = rect;
	begin_info.framebuffer           = p_device->m_pipeline_allocator->CreateFrameBuffer(frame_buffer);
	begin_info.clearValueCount       = static_cast<uint32_t>(frame_buffer.m_clear_values.size());
	begin_info.pClearValues          = frame_buffer.m_clear_values.data();

	vkCmdBeginRenderPass(m_handle, &begin_info, VK_SUBPASS_CONTENTS_INLINE);
}

void CommandBuffer::EndRenderPass()
{
	vkCmdEndRenderPass(m_handle);
}

void CommandBuffer::Bind(PipelineState &pso)
{
	m_current_pso = &pso;
	vkCmdBindPipeline(m_handle, pso.GetBindPoint(), p_device->m_pipeline_allocator->CreatePipeline(*m_current_pso));
}

void CommandBuffer::Bind(DescriptorState &descriptor_state)
{
	ASSERT(m_current_pso);
	descriptor_state.Write();
	for (auto& [set, descriptor_set] : descriptor_state.m_descriptor_sets)
	{
		vkCmdBindDescriptorSets(m_handle, m_current_pso->GetBindPoint(), p_device->m_pipeline_allocator->CreatePipelineLayout(*m_current_pso), set, 1, &descriptor_set, 0, nullptr);
	}
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
	barrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.srcAccessMask        = src.access_mask;
	barrier.dstAccessMask        = dst.access_mask;
	barrier.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
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
		buffer_barriers[i].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		buffer_barriers[i].srcAccessMask       = buffer_transitions[i].src.access_mask;
		buffer_barriers[i].dstAccessMask       = buffer_transitions[i] .dst.access_mask;
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
		image_barriers[i].sType       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		image_barriers[i].srcAccessMask        = texture_transitions[i].src.access_mask;
		image_barriers[i].dstAccessMask        = texture_transitions[i].dst.access_mask;
		image_barriers[i].oldLayout            = texture_transitions[i].src.layout;
		image_barriers[i].newLayout            = texture_transitions[i].dst.layout;
		image_barriers[i].srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
		image_barriers[i].dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
		image_barriers[i].image                = *texture_transitions[i].texture;
		image_barriers[i].subresourceRange     = texture_transitions[i].range;
		src_stage |= texture_transitions[i].src.stage;
		dst_stage |= texture_transitions[i].dst.stage;
	}

	vkCmdPipelineBarrier(*this, src_stage, dst_stage, 0, 0, nullptr, static_cast<uint32_t>(buffer_barriers.size()), buffer_barriers.data(), static_cast<uint32_t>(image_barriers.size()), image_barriers.data());
}

CommandBuffer::operator const VkCommandBuffer &() const
{
	return m_handle;
}
}        // namespace Ilum