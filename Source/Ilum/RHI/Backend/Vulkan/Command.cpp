#include "Command.hpp"
#include "Buffer.hpp"
#include "Descriptor.hpp"
#include "Device.hpp"
#include "PipelineState.hpp"
#include "RenderTarget.hpp"
#include "Texture.hpp"

namespace Ilum::Vulkan
{
static std::unordered_map<size_t, VkCommandPool>          CommandPools;
static std::unordered_map<size_t, std::vector<Command *>> CommandBuffers;

static uint32_t CommandCount = 0;

Command::Command(RHIDevice *device, RHIQueueFamily family) :
    RHICommand(device, family)
{
	CommandCount++;

	// Register Command Buffer
	size_t hash = 0;
	HashCombine(hash, family, std::this_thread::get_id());
	CommandBuffers[hash].push_back(this);

	// Check Command Pool
	if (CommandPools.find(hash) == CommandPools.end())
	{
		VkCommandPoolCreateInfo create_info = {};
		create_info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		create_info.flags                   = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
		create_info.queueFamilyIndex        = static_cast<Device *>(p_device)->GetQueueFamily(family);

		VkCommandPool pool = VK_NULL_HANDLE;
		vkCreateCommandPool(static_cast<Device *>(p_device)->GetDevice(), &create_info, nullptr, &pool);
		CommandPools[hash] = pool;
	}
	m_pool = CommandPools[hash];

	// Create Command Buffer
	VkCommandBufferAllocateInfo allocate_info = {};
	allocate_info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocate_info.commandPool                 = m_pool;
	allocate_info.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocate_info.commandBufferCount          = 1;
	vkAllocateCommandBuffers(static_cast<Device *>(p_device)->GetDevice(), &allocate_info, &m_handle);
}

Command::Command(RHIDevice *device, VkCommandPool pool, RHIQueueFamily family) :
    RHICommand(device, family), m_pool(pool)
{
	CommandCount++;

	VkCommandBufferAllocateInfo allocate_info = {};
	allocate_info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocate_info.commandPool                 = pool;
	allocate_info.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocate_info.commandBufferCount          = 1;
	vkAllocateCommandBuffers(static_cast<Device *>(p_device)->GetDevice(), &allocate_info, &m_handle);
}

Command::~Command()
{
	if (m_handle)
	{
		vkFreeCommandBuffers(static_cast<Device *>(p_device)->GetDevice(), m_pool, 1, &m_handle);
	}

	if (--CommandCount == 0)
	{
		for (auto &[hash, pool] : CommandPools)
		{
			vkDestroyCommandPool(static_cast<Device *>(p_device)->GetDevice(), pool, nullptr);
		}
		CommandPools.clear();
	}

	for (auto &[hash, cmds] : CommandBuffers)
	{
		for (auto iter = cmds.begin(); iter != cmds.end(); iter++)
		{
			if (*iter == this)
			{
				cmds.erase(iter);
				return;
			}
		}
	}
}

void Command::SetState(CommandState state)
{
	m_state = state;
}

VkCommandBuffer Command::GetHandle() const
{
	return m_handle;
}

void Command::Begin()
{
	ASSERT(m_state == CommandState::Initial);

	VkCommandBufferBeginInfo begin_info = {};
	begin_info.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin_info.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	begin_info.pInheritanceInfo         = nullptr;

	vkBeginCommandBuffer(m_handle, &begin_info);

	m_state = CommandState::Recording;
}

void Command::End()
{
	ASSERT(m_state == CommandState::Recording);
	vkEndCommandBuffer(m_handle);
	m_state = CommandState::Executable;

	p_descriptor = nullptr;
}

void Command::BeginRenderPass(RHIRenderTarget *render_target)
{
	if (static_cast<Device *>(p_device)->IsFeatureSupport(VulkanFeature::DynamicRendering))
	{
		VkRenderingInfo rendering_info = {};
		rendering_info.sType           = VK_STRUCTURE_TYPE_RENDERING_INFO;
		rendering_info.renderArea      = {0, 0, render_target->GetWidth(), render_target->GetHeight()};
		rendering_info.layerCount      = render_target->GetLayers();

		RenderTarget *vk_render_target = static_cast<RenderTarget *>(render_target);

		rendering_info.colorAttachmentCount = static_cast<uint32_t>(vk_render_target->GetColorAttachments().size());
		rendering_info.pColorAttachments    = vk_render_target->GetColorAttachments().data();
		rendering_info.pDepthAttachment     = vk_render_target->GetDepthAttachment().has_value() ? &vk_render_target->GetDepthAttachment().value() : nullptr;
		rendering_info.pStencilAttachment   = vk_render_target->GetStencilAttachment().has_value() ? &vk_render_target->GetStencilAttachment().value() : nullptr;

		vkCmdBeginRendering(m_handle, &rendering_info);
	}
	else
	{
		RenderTarget *vk_render_target = static_cast<RenderTarget *>(render_target);

		auto clear_values = vk_render_target->GetClearValue();

		VkRenderPassBeginInfo begin_info = {};
		begin_info.sType                 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		begin_info.framebuffer           = vk_render_target->GetFramebuffer();
		begin_info.pClearValues          = clear_values.data();
		begin_info.clearValueCount       = static_cast<uint32_t>(clear_values.size());
		begin_info.renderPass            = vk_render_target->GetRenderPass();
		begin_info.renderArea            = vk_render_target->GetRenderArea();

		vkCmdBeginRenderPass(m_handle, &begin_info, VK_SUBPASS_CONTENTS_INLINE);
	}

	p_render_target = static_cast<RenderTarget *>(render_target);
}

void Command::EndRenderPass()
{
	if (static_cast<Device *>(p_device)->IsFeatureSupport(VulkanFeature::DynamicRendering))
	{
		vkCmdEndRendering(m_handle);
	}
	else
	{
		vkCmdEndRenderPass(m_handle);
	}

	p_render_target = nullptr;
}

void Command::BindVertexBuffer(RHIBuffer *vertex_buffer)
{
	Buffer  *vk_vertex_buffer = static_cast<Buffer *>(vertex_buffer);
	size_t   offset           = 0;
	VkBuffer handle           = vk_vertex_buffer->GetHandle();
	vkCmdBindVertexBuffers(m_handle, 0, 1, &handle, &offset);
}

void Command::BindIndexBuffer(RHIBuffer *index_buffer, bool is_short)
{
	vkCmdBindIndexBuffer(m_handle, static_cast<Buffer *>(index_buffer)->GetHandle(), 0, is_short ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
}

void Command::BindDescriptor(RHIDescriptor *descriptor)
{
	p_descriptor = static_cast<Descriptor *>(descriptor);
}

void Command::BindPipelineState(RHIPipelineState *pipeline_state)
{
	ASSERT(p_descriptor != nullptr);
	PipelineState *pso = static_cast<PipelineState *>(pipeline_state);
	vkCmdBindPipeline(m_handle, pso->GetPipelineBindPoint(), pso->GetPipeline(p_descriptor, p_render_target));
	for (auto &[set, descriptor_set] : p_descriptor->GetDescriptorSet())
	{
		vkCmdBindDescriptorSets(m_handle, pso->GetPipelineBindPoint(), pso->GetPipelineLayout(p_descriptor), set, 1, &descriptor_set, 0, nullptr);
	}
}

void Command::SetViewport(float width, float height, float x, float y)
{
	// Flip y
	VkViewport viewport = {x, y, width, height, 0.f, 1.f};
	vkCmdSetViewport(m_handle, 0, 1, &viewport);
}

void Command::SetScissor(uint32_t width, uint32_t height, int32_t offset_x, int32_t offset_y)
{
	VkRect2D rect = {VkOffset2D{offset_x, offset_y}, VkExtent2D{width, height}};
	vkCmdSetScissor(m_handle, 0, 1, &rect);
}

void Command::Dispatch(uint32_t group_x, uint32_t group_y, uint32_t group_z)
{
	vkCmdDispatch(m_handle, group_x, group_y, group_z);
}

void Command::Draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance)
{
	vkCmdDraw(m_handle, vertex_count, instance_count, first_vertex, first_instance);
}

void Command::DrawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, uint32_t vertex_offset, uint32_t first_instance)
{
	vkCmdDrawIndexed(m_handle, index_count, instance_count, first_index, vertex_offset, first_instance);
}

void Command::CopyBufferToTexture(RHIBuffer *src_buffer, RHITexture *dst_texture, uint32_t mip_level, uint32_t base_layer, uint32_t layer_count)
{
	VkImageSubresourceLayers subresource = {};
	subresource.aspectMask               = IsDepthFormat(dst_texture->GetDesc().format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
	subresource.baseArrayLayer           = base_layer;
	subresource.mipLevel                 = mip_level;
	subresource.layerCount               = layer_count;

	uint32_t width  = std::max(dst_texture->GetDesc().width, 1u << mip_level) >> mip_level;
	uint32_t height = std::max(dst_texture->GetDesc().height, 1u << mip_level) >> mip_level;

	VkBufferImageCopy copy_info = {};
	copy_info.bufferOffset      = 0;
	copy_info.bufferImageHeight = 0;
	copy_info.bufferRowLength   = 0;
	copy_info.imageSubresource  = subresource;
	copy_info.imageOffset       = {0, 0, 0};
	copy_info.imageExtent       = {width, height, 1};

	vkCmdCopyBufferToImage(m_handle, static_cast<Buffer *>(src_buffer)->GetHandle(), static_cast<Texture *>(dst_texture)->GetHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_info);
}

void Command::ResourceStateTransition(const std::vector<TextureStateTransition> &texture_transitions, const std::vector<BufferStateTransition> &buffer_transitions)
{
	std::vector<VkBufferMemoryBarrier> buffer_barriers(buffer_transitions.size());
	std::vector<VkImageMemoryBarrier>  image_barriers(texture_transitions.size());

	VkPipelineStageFlags src_stage = 0;
	VkPipelineStageFlags dst_stage = 0;

	for (uint32_t i = 0; i < buffer_barriers.size(); i++)
	{
		BufferState vk_buffer_state_src = BufferState::Create(buffer_transitions[i].src);
		BufferState vk_buffer_state_dst = BufferState::Create(buffer_transitions[i].dst);
		if (vk_buffer_state_src == vk_buffer_state_dst)
		{
			continue;
		}

		buffer_barriers[i].sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		buffer_barriers[i].srcAccessMask       = vk_buffer_state_src.access_mask;
		buffer_barriers[i].dstAccessMask       = vk_buffer_state_dst.access_mask;
		buffer_barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		buffer_barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		buffer_barriers[i].buffer              = static_cast<Buffer *>(buffer_transitions[i].buffer)->GetHandle();
		buffer_barriers[i].offset              = 0;
		buffer_barriers[i].size                = buffer_transitions[i].buffer->GetDesc().size;
		src_stage |= vk_buffer_state_src.stage;
		dst_stage |= vk_buffer_state_dst.stage;
	}

	for (uint32_t i = 0; i < image_barriers.size(); i++)
	{
		TextureState vk_texture_state_src = TextureState::Create(texture_transitions[i].src);
		TextureState vk_texture_state_dst = TextureState::Create(texture_transitions[i].dst);
		if (vk_texture_state_src == vk_texture_state_dst)
		{
			continue;
		}

		VkImageSubresourceRange range = {};
		range.aspectMask              = IsDepthFormat(texture_transitions[i].texture->GetDesc().format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
		range.baseArrayLayer          = texture_transitions[i].range.base_layer;
		range.baseMipLevel            = texture_transitions[i].range.base_mip;
		range.layerCount              = texture_transitions[i].range.layer_count;
		range.levelCount              = texture_transitions[i].range.mip_count;

		image_barriers[i].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		image_barriers[i].srcAccessMask       = vk_texture_state_src.access_mask;
		image_barriers[i].dstAccessMask       = vk_texture_state_dst.access_mask;
		image_barriers[i].oldLayout           = vk_texture_state_src.layout;
		image_barriers[i].newLayout           = vk_texture_state_dst.layout;
		image_barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		image_barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		image_barriers[i].image               = static_cast<Texture *>(texture_transitions[i].texture)->GetHandle();
		image_barriers[i].subresourceRange    = range;
		src_stage |= vk_texture_state_src.stage;
		dst_stage |= vk_texture_state_dst.stage;
	}

	vkCmdPipelineBarrier(m_handle, src_stage, dst_stage, 0, 0, nullptr, static_cast<uint32_t>(buffer_barriers.size()), buffer_barriers.data(), static_cast<uint32_t>(image_barriers.size()), image_barriers.data());
}
}        // namespace Ilum::Vulkan