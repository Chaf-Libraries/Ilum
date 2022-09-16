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

void Command::BeginMarker(const std::string &name, float r, float g, float b, float a)
{
	VkDebugUtilsLabelEXT label = {};
	label.sType                = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
	label.pLabelName           = name.c_str();
	label.color[0]             = r;
	label.color[1]             = g;
	label.color[2]             = b;
	label.color[3]             = a;

	static_cast<Device *>(p_device)->BeginDebugUtilsLabel(m_handle, label);
}

void Command::EndMarker()
{
	static_cast<Device *>(p_device)->EndDebugUtilsLabel(m_handle);
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
	p_pipeline_state = static_cast<PipelineState *>(pipeline_state);
	vkCmdBindPipeline(m_handle, p_pipeline_state->GetPipelineBindPoint(), p_pipeline_state->GetPipeline(p_descriptor, p_render_target));
	for (auto &[set, descriptor_set] : p_descriptor->GetDescriptorSet())
	{
		vkCmdBindDescriptorSets(m_handle, p_pipeline_state->GetPipelineBindPoint(), p_pipeline_state->GetPipelineLayout(p_descriptor), set, 1, &descriptor_set, 0, nullptr);
	}
	for (auto &[name, constant] : static_cast<Descriptor *>(p_descriptor)->GetConstantResolve())
	{
		vkCmdPushConstants(m_handle, p_pipeline_state->GetPipelineLayout(p_descriptor), constant.stage, constant.offset, static_cast<uint32_t>(constant.data.size()), constant.data.data());
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

void Command::Dispatch(uint32_t thread_x, uint32_t thread_y, uint32_t thread_z, uint32_t block_x, uint32_t block_y, uint32_t block_z)
{
	vkCmdDispatch(m_handle, (thread_x + block_x - 1) / block_x, (thread_y + block_y - 1) / block_y, (thread_z + block_z - 1) / block_z);
}

void Command::Draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance)
{
	vkCmdDraw(m_handle, vertex_count, instance_count, first_vertex, first_instance);
}

void Command::DrawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, uint32_t vertex_offset, uint32_t first_instance)
{
	vkCmdDrawIndexed(m_handle, index_count, instance_count, first_index, vertex_offset, first_instance);
}

void Command::DrawMeshTask(uint32_t thread_x, uint32_t thread_y, uint32_t thread_z, uint32_t block_x, uint32_t block_y, uint32_t block_z)
{
	vkCmdDrawMeshTasksEXT(m_handle, (thread_x + block_x - 1) / block_x, (thread_y + block_y - 1) / block_y, (thread_z + block_z - 1) / block_z);
}

void Command::TraceRay(uint32_t width, uint32_t height, uint32_t depth)
{
	auto sbt = p_pipeline_state->GetShaderBindingTable(p_pipeline_state->GetPipeline(p_descriptor, p_render_target));
	vkCmdTraceRaysKHR(
	    m_handle,
	    sbt.raygen, sbt.miss, sbt.hit, sbt.callable,
	    width, height, depth);
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

void Command::CopyTextureToBuffer(RHITexture *src_texture, RHIBuffer *dst_buffer, uint32_t mip_level, uint32_t base_layer, uint32_t layer_count)
{
	VkImageSubresourceLayers subresource = {};
	subresource.aspectMask               = IsDepthFormat(src_texture->GetDesc().format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
	subresource.baseArrayLayer           = base_layer;
	subresource.mipLevel                 = mip_level;
	subresource.layerCount               = layer_count;

	uint32_t width  = std::max(src_texture->GetDesc().width, 1u << mip_level) >> mip_level;
	uint32_t height = std::max(src_texture->GetDesc().height, 1u << mip_level) >> mip_level;

	VkBufferImageCopy copy_info = {};
	copy_info.bufferOffset      = 0;
	copy_info.bufferImageHeight = width;
	copy_info.bufferRowLength   = height;
	copy_info.imageSubresource  = subresource;
	copy_info.imageOffset       = {0, 0, 0};
	copy_info.imageExtent       = {width, height, 1};

	vkCmdCopyImageToBuffer(m_handle, static_cast<Texture *>(src_texture)->GetHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, static_cast<Buffer *>(dst_buffer)->GetHandle(), 1, &copy_info);
}

void Command::CopyBufferToBuffer(RHIBuffer *src_buffer, RHIBuffer *dst_buffer, size_t size, size_t src_offset, size_t dst_offset)
{
	VkBufferCopy buffer_copy = {};
	buffer_copy.size         = size;
	buffer_copy.srcOffset    = src_offset;
	buffer_copy.dstOffset    = dst_offset;

	vkCmdCopyBuffer(m_handle, static_cast<Buffer *>(src_buffer)->GetHandle(), static_cast<Buffer *>(dst_buffer)->GetHandle(), 1, &buffer_copy);
}

void Command::GenerateMipmaps(RHITexture *texture, RHIResourceState initial_state, RHIFilter filter)
{
	TextureRange             src_range  = {RHITextureDimension::Texture2D, 0, texture->GetDesc().mips, 0, texture->GetDesc().layers};
	TextureRange             dst_range  = {RHITextureDimension::Texture2D, 0, texture->GetDesc().mips, 0, texture->GetDesc().layers};
	VkImageSubresourceLayers src_layers = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	VkImageSubresourceLayers dst_layers = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	RHIResourceState         src_state  = RHIResourceState::TransferDest;

	uint32_t src_width  = texture->GetDesc().width;
	uint32_t src_height = texture->GetDesc().height;
	uint32_t dst_width  = texture->GetDesc().width;
	uint32_t dst_height = texture->GetDesc().height;

	for (uint32_t i = 1; i < texture->GetDesc().mips; i++)
	{
		src_width  = dst_width;
		src_height = dst_height;

		dst_width  = std::max(src_width / 2, 1u);
		dst_height = std::max(src_height / 2, 1u);

		src_layers.mipLevel = i - 1;
		src_range.base_mip  = i - 1;
		src_range.mip_count = 1;

		dst_layers.mipLevel = i;
		dst_range.base_mip  = i;
		dst_range.mip_count = 1;

		std::vector<TextureStateTransition> transitions(2);
		transitions[0].texture = texture;
		transitions[0].src     = src_state;
		transitions[0].dst     = RHIResourceState::TransferSource;
		transitions[0].range   = src_range;
		transitions[1].texture = texture;
		transitions[1].src     = RHIResourceState::Undefined;
		transitions[1].dst     = RHIResourceState::TransferDest;
		transitions[1].range   = dst_range;
		ResourceStateTransition(transitions, {});

		src_state = RHIResourceState::TransferDest;

		VkImageBlit blit_info    = {};
		blit_info.srcOffsets[0]  = {0, 0, 0};
		blit_info.srcOffsets[1]  = {static_cast<int32_t>(src_width), static_cast<int32_t>(src_height), 1};
		blit_info.dstOffsets[0]  = {0, 0, 0};
		blit_info.dstOffsets[1]  = {static_cast<int32_t>(dst_width), static_cast<int32_t>(dst_height), 1};
		blit_info.srcSubresource = src_layers;
		blit_info.dstSubresource = dst_layers;

		vkCmdBlitImage(m_handle, static_cast<Texture *>(texture)->GetHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, static_cast<Texture *>(texture)->GetHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit_info, ToVulkanFilter[filter]);
	}

	TextureRange mip_level_range = {RHITextureDimension::Texture2D, 0, texture->GetDesc().mips, 0, texture->GetDesc().layers};

	mip_level_range.mip_count = mip_level_range.mip_count - 1;
	ResourceStateTransition({TextureStateTransition{texture, RHIResourceState::TransferSource, RHIResourceState::TransferDest, mip_level_range}}, {});
}

void Command::BlitTexture(RHITexture *src_texture, const TextureRange &src_range, const RHIResourceState &src_state, RHITexture *dst_texture, const TextureRange &dst_range, const RHIResourceState &dst_state, RHIFilter filter)
{
	VkImage       src_image  = static_cast<Texture *>(src_texture)->GetHandle();
	VkImage       dst_image  = static_cast<Texture *>(dst_texture)->GetHandle();
	VkImageLayout src_layout = TextureState::Create(src_state).layout;
	VkImageLayout dst_layout = TextureState::Create(dst_state).layout;

	VkImageBlit region = {};

	region.srcSubresource.aspectMask     = IsDepthFormat(src_texture->GetDesc().format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
	region.srcSubresource.baseArrayLayer = src_range.base_layer;
	region.srcSubresource.layerCount     = src_range.layer_count;
	region.srcSubresource.mipLevel       = src_range.base_mip;

	region.dstSubresource.aspectMask     = IsDepthFormat(dst_texture->GetDesc().format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
	region.dstSubresource.baseArrayLayer = dst_range.base_layer;
	region.dstSubresource.layerCount     = dst_range.layer_count;
	region.dstSubresource.mipLevel       = dst_range.base_mip;

	region.srcOffsets[1].x = src_texture->GetDesc().width;
	region.srcOffsets[1].y = src_texture->GetDesc().height;
	region.srcOffsets[1].z = 1;

	region.dstOffsets[1].x = dst_texture->GetDesc().width;
	region.dstOffsets[1].y = dst_texture->GetDesc().height;
	region.dstOffsets[1].z = 1;

	vkCmdBlitImage(m_handle, src_image, src_layout, dst_image, dst_layout, 1, &region, ToVulkanFilter[filter]);
}

void Command::ResourceStateTransition(const std::vector<TextureStateTransition> &texture_transitions, const std::vector<BufferStateTransition> &buffer_transitions)
{
	std::vector<VkBufferMemoryBarrier> buffer_barriers;
	buffer_barriers.reserve(buffer_transitions.size());

	std::vector<VkImageMemoryBarrier> image_barriers;
	image_barriers.reserve(texture_transitions.size());

	VkPipelineStageFlags src_stage = 0;
	VkPipelineStageFlags dst_stage = 0;

	for (uint32_t i = 0; i < buffer_transitions.size(); i++)
	{
		BufferState vk_buffer_state_src = BufferState::Create(buffer_transitions[i].src);
		BufferState vk_buffer_state_dst = BufferState::Create(buffer_transitions[i].dst);
		if (vk_buffer_state_src == vk_buffer_state_dst)
		{
			continue;
		}

		VkBufferMemoryBarrier buffer_barrier = {};

		buffer_barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		buffer_barrier.srcAccessMask       = vk_buffer_state_src.access_mask;
		buffer_barrier.dstAccessMask       = vk_buffer_state_dst.access_mask;
		buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		buffer_barrier.buffer              = static_cast<Buffer *>(buffer_transitions[i].buffer)->GetHandle();
		buffer_barrier.offset              = 0;
		buffer_barrier.size                = buffer_transitions[i].buffer->GetDesc().size;
		buffer_barriers.push_back(buffer_barrier);
		src_stage |= vk_buffer_state_src.stage;
		dst_stage |= vk_buffer_state_dst.stage;
	}

	for (uint32_t i = 0; i < texture_transitions.size(); i++)
	{
		TextureState vk_texture_state_src = TextureState::Create(texture_transitions[i].src);
		TextureState vk_texture_state_dst = TextureState::Create(texture_transitions[i].dst);
		if (vk_texture_state_src == vk_texture_state_dst)
		{
			continue;
		}

		VkImageMemoryBarrier image_barrier = {};

		VkImageSubresourceRange range = {};
		range.aspectMask              = IsDepthFormat(texture_transitions[i].texture->GetDesc().format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
		range.baseArrayLayer          = texture_transitions[i].range.base_layer;
		range.baseMipLevel            = texture_transitions[i].range.base_mip;
		range.layerCount              = texture_transitions[i].range.layer_count;
		range.levelCount              = texture_transitions[i].range.mip_count;

		image_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		image_barrier.srcAccessMask       = vk_texture_state_src.access_mask;
		image_barrier.dstAccessMask       = vk_texture_state_dst.access_mask;
		image_barrier.oldLayout           = vk_texture_state_src.layout;
		image_barrier.newLayout           = vk_texture_state_dst.layout;
		image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		image_barrier.image               = static_cast<Texture *>(texture_transitions[i].texture)->GetHandle();
		image_barrier.subresourceRange    = range;
		image_barriers.push_back(image_barrier);
		src_stage |= vk_texture_state_src.stage;
		dst_stage |= vk_texture_state_dst.stage;
	}

	vkCmdPipelineBarrier(m_handle, src_stage, dst_stage, 0, 0, nullptr, static_cast<uint32_t>(buffer_barriers.size()), buffer_barriers.data(), static_cast<uint32_t>(image_barriers.size()), image_barriers.data());
}
}        // namespace Ilum::Vulkan