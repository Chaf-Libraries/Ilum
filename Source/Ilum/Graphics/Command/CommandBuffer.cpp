#include "CommandBuffer.hpp"
#include "CommandPool.hpp"

#include "Device/LogicalDevice.hpp"

#include "Engine/Context.hpp"
#include "Engine/Engine.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Synchronization/Queue.hpp"
#include "Graphics/Synchronization/QueueSystem.hpp"
#include "Graphics/Vulkan/Vulkan.hpp"

namespace Ilum
{
CommandBuffer::CommandBuffer(CommandPool &cmd_pool, QueueUsage usage, VkCommandBufferLevel level) :
    m_command_pool(cmd_pool)
{
	VkCommandBufferAllocateInfo command_buffer_allocate_info = {};
	command_buffer_allocate_info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	command_buffer_allocate_info.commandPool                 = m_command_pool;
	command_buffer_allocate_info.level                       = level;
	command_buffer_allocate_info.commandBufferCount          = 1;
	if (!VK_CHECK(vkAllocateCommandBuffers(Engine::instance()->getContext().getSubsystem<GraphicsContext>()->getLogicalDevice(), &command_buffer_allocate_info, &m_handle)))
	{
		VK_ERROR("Failed to create command buffer!");
		return;
	}
}

CommandBuffer::CommandBuffer(QueueUsage usage, VkCommandBufferLevel level):
    m_command_pool(GraphicsContext::instance()->getCommandPool(usage, CommandPool::ResetMode::AlwaysAllocate))
{
	VkCommandBufferAllocateInfo command_buffer_allocate_info = {};
	command_buffer_allocate_info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	command_buffer_allocate_info.commandPool                 = m_command_pool;
	command_buffer_allocate_info.level                       = level;
	command_buffer_allocate_info.commandBufferCount          = 1;
	if (!VK_CHECK(vkAllocateCommandBuffers(Engine::instance()->getContext().getSubsystem<GraphicsContext>()->getLogicalDevice(), &command_buffer_allocate_info, &m_handle)))
	{
		VK_ERROR("Failed to create command buffer!");
		return;
	}
}

CommandBuffer::~CommandBuffer()
{
	if (m_handle)
	{
		vkFreeCommandBuffers(GraphicsContext::instance()->getLogicalDevice(), m_command_pool, 1, &m_handle);
	}
}

void CommandBuffer::reset() const
{
	if (m_command_pool.getResetMode() == CommandPool::ResetMode::ResetIndividually)
	{
		if (!VK_CHECK(vkResetCommandBuffer(m_handle, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT)))
		{
			VK_ERROR("Failed to reset command buffer!");
			return;
		}
	}
}

bool CommandBuffer::begin(VkCommandBufferUsageFlagBits usage, VkCommandBufferInheritanceInfo *inheritanceInfo) const
{
	VkCommandBufferBeginInfo command_buffer_begin_info = {};
	command_buffer_begin_info.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	command_buffer_begin_info.flags                    = usage;
	command_buffer_begin_info.pInheritanceInfo         = inheritanceInfo;

	if (!VK_CHECK(vkBeginCommandBuffer(m_handle, &command_buffer_begin_info)))
	{
		VK_ERROR("Failed to begin command buffer!");
		return false;
	}
	return true;
}

bool CommandBuffer::beginRenderPass(const PassNative &pass) const
{
	if (!pass.render_pass)
	{
		return false;
	}

	VkRenderPassBeginInfo begin_info = {};
	begin_info.sType                 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	begin_info.renderPass            = pass.render_pass;
	begin_info.renderArea            = pass.render_area;
	begin_info.framebuffer           = pass.frame_buffer;
	begin_info.clearValueCount       = static_cast<uint32_t>(pass.clear_values.size());
	begin_info.pClearValues          = pass.clear_values.data();

	vkCmdBeginRenderPass(*this, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

	if (pass.pipeline)
	{
		vkCmdBindPipeline(*this, pass.bind_point, pass.pipeline);
	}
	for (auto &descriptor_set : pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(*this, pass.bind_point, pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	return true;
}

void CommandBuffer::endRenderPass() const
{
	vkCmdEndRenderPass(*this);
}

void CommandBuffer::end() const
{
	if (!VK_CHECK(vkEndCommandBuffer(m_handle)))
	{
		VK_ERROR("Failed to end command buffer!");
		return;
	}
}

void CommandBuffer::copyImage(const ImageInfo &src, const ImageInfo &dst) const
{
	auto src_range = src.resource.get().getSubresourceRange();
	auto dst_range = dst.resource.get().getSubresourceRange();

	std::array<VkImageMemoryBarrier, 2> barriers = {};

	uint32_t barrier_count = 0;

	if (src.usage != VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
	{
		auto &src_barrier               = barriers[barrier_count++];
		src_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		src_barrier.srcAccessMask       = Image::usage_to_access(src.usage);
		src_barrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
		src_barrier.oldLayout           = Image::usage_to_layout(src.usage);
		src_barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		src_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		src_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		src_barrier.image               = src.resource.get();
		src_barrier.subresourceRange    = src_range;
	}
	if (dst.usage != VK_IMAGE_USAGE_TRANSFER_DST_BIT)
	{
		auto &dst_barrier               = barriers[barrier_count++];
		dst_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		dst_barrier.srcAccessMask       = Image::usage_to_access(dst.usage);
		dst_barrier.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
		dst_barrier.oldLayout           = Image::usage_to_layout(dst.usage);
		dst_barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		dst_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		dst_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		dst_barrier.image               = dst.resource.get();
		dst_barrier.subresourceRange    = dst_range;
	}

	if (barrier_count > 0)
	{
		vkCmdPipelineBarrier(*this, Image::usage_to_stage(src.usage) | Image::usage_to_stage(dst.usage), VK_PIPELINE_STAGE_TRANSFER_BIT, {}, 0, nullptr, 0, nullptr, barrier_count, barriers.data());
	}

	auto src_layers = src.resource.get().getSubresourceLayers(src.mip_level, src.layer);
	auto dst_layers = dst.resource.get().getSubresourceLayers(dst.mip_level, dst.layer);

	VkImageCopy copy_info    = {};
	copy_info.srcOffset      = {0, 0, 0};
	copy_info.dstOffset      = {0, 0, 0};
	copy_info.srcSubresource = src_layers;
	copy_info.dstSubresource = dst_layers;
	copy_info.extent         = {dst.resource.get().getMipWidth(dst.mip_level),
                        dst.resource.get().getMipHeight(dst.mip_level), 1};

	vkCmdCopyImage(*this, src.resource.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.resource.get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_info);
}

void CommandBuffer::copyBufferToImage(const BufferInfo &src, const ImageInfo &dst) const
{
	if (dst.usage != VK_IMAGE_USAGE_TRANSFER_DST_BIT)
	{
		auto                 dst_range = dst.resource.get().getSubresourceRange();
		VkImageMemoryBarrier barrier   = {};
		barrier.sType                  = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.srcAccessMask          = Image::usage_to_access(dst.usage);
		barrier.dstAccessMask          = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.oldLayout              = Image::usage_to_layout(dst.usage);
		barrier.newLayout              = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.srcQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
		barrier.image                  = dst.resource.get();
		barrier.subresourceRange       = dst_range;

		vkCmdPipelineBarrier(*this, Image::usage_to_stage(dst.usage), VK_PIPELINE_STAGE_TRANSFER_BIT, {}, 0, nullptr, 0, nullptr, 1, &barrier);
	}

	auto dst_layer = dst.resource.get().getSubresourceLayers(dst.mip_level, dst.layer);

	VkBufferImageCopy copy_info = {};
	copy_info.bufferOffset      = src.offset;
	copy_info.bufferImageHeight = 0;
	copy_info.bufferRowLength   = 0;
	copy_info.imageSubresource  = dst_layer;
	copy_info.imageOffset       = {0, 0, 0};
	copy_info.imageExtent       = {dst.resource.get().getMipWidth(dst.mip_level),
                             dst.resource.get().getMipHeight(dst.mip_level), 1};

	vkCmdCopyBufferToImage(*this, src.resource.get(), dst.resource.get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_info);
}

void CommandBuffer::copyImageToBuffer(const ImageInfo &src, const BufferInfo &dst) const
{
	if (src.usage != VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
	{
		auto                 src_range = src.resource.get().getSubresourceRange();
		VkImageMemoryBarrier barrier   = {};
		barrier.sType                  = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.srcAccessMask          = Image::usage_to_access(src.usage);
		barrier.dstAccessMask          = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.oldLayout              = Image::usage_to_layout(src.usage);
		barrier.newLayout              = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.srcQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
		barrier.image                  = src.resource.get();
		barrier.subresourceRange       = src_range;

		vkCmdPipelineBarrier(*this, Image::usage_to_stage(src.usage), VK_PIPELINE_STAGE_TRANSFER_BIT, {}, 0, nullptr, 0, nullptr, 1, &barrier);
	}

	auto src_layer = src.resource.get().getSubresourceLayers(src.mip_level, src.layer);

	VkBufferImageCopy copy_info = {};
	copy_info.bufferOffset      = dst.offset;
	copy_info.bufferImageHeight = src.resource.get().getMipHeight(src.mip_level);
	copy_info.bufferRowLength   = src.resource.get().getMipWidth(src.mip_level);
	copy_info.imageSubresource  = src_layer;
	copy_info.imageOffset       = {0, 0, 0};
	copy_info.imageExtent       = {src.resource.get().getMipWidth(src.mip_level),
                             src.resource.get().getMipHeight(src.mip_level), 1};

	vkCmdCopyImageToBuffer(*this, src.resource.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.resource.get(), 1, &copy_info);
}

void CommandBuffer::copyBuffer(const BufferInfo &src, const BufferInfo &dst, VkDeviceSize size) const
{
	ASSERT(src.resource.get().getSize() >= src.offset + size);
	ASSERT(dst.resource.get().getSize() >= dst.offset + size);

	VkBufferCopy copy_info = {};
	copy_info.dstOffset    = dst.offset;
	copy_info.size         = size;
	copy_info.srcOffset    = src.offset;

	vkCmdCopyBuffer(*this, src.resource.get(), dst.resource.get(), 1, &copy_info);
}

void CommandBuffer::blitImage(const Image &src, VkImageUsageFlagBits src_usage, const Image &dst, VkImageUsageFlagBits dst_usage, VkFilter filter) const
{
	auto src_range = src.getSubresourceRange();
	auto dst_range = dst.getSubresourceRange();

	std::array<VkImageMemoryBarrier, 2> barriers = {};

	uint32_t barrier_count = 0;

	if (src_usage != VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
	{
		auto &src_barrier               = barriers[barrier_count++];
		src_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		src_barrier.srcAccessMask       = Image::usage_to_access(src_usage);
		src_barrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
		src_barrier.oldLayout           = Image::usage_to_layout(src_usage);
		src_barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		src_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		src_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		src_barrier.image               = src;
		src_barrier.subresourceRange    = src_range;
	}
	if (dst_usage != VK_IMAGE_USAGE_TRANSFER_DST_BIT)
	{
		auto &dst_barrier               = barriers[barrier_count++];
		dst_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		dst_barrier.srcAccessMask       = Image::usage_to_access(dst_usage);
		dst_barrier.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
		dst_barrier.oldLayout           = Image::usage_to_layout(dst_usage);
		dst_barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		dst_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		dst_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		dst_barrier.image               = dst;
		dst_barrier.subresourceRange    = dst_range;
	}

	if (barrier_count > 0)
	{
		vkCmdPipelineBarrier(*this, Image::usage_to_stage(src_usage) | Image::usage_to_stage(dst_usage), VK_PIPELINE_STAGE_TRANSFER_BIT, {}, 0, nullptr, 0, nullptr, barrier_count, barriers.data());
	}

	auto src_layers = src.getSubresourceLayers();
	auto dst_layers = dst.getSubresourceLayers();

	VkImageBlit blit_info    = {};
	blit_info.srcOffsets[0]  = {0, 0, 0};
	blit_info.srcOffsets[1]  = {static_cast<int32_t>(src.getWidth()), static_cast<int32_t>(src.getHeight()), 1};
	blit_info.dstOffsets[0]  = {0, 0, 0};
	blit_info.dstOffsets[1]  = {static_cast<int32_t>(src.getWidth()), static_cast<int32_t>(src.getHeight()), 1};
	blit_info.srcSubresource = src_layers;
	blit_info.dstSubresource = dst_layers;

	vkCmdBlitImage(*this, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit_info, filter);
}

void CommandBuffer::generateMipmaps(const Image &image, VkImageUsageFlagBits initial_usage, VkFilter filter) const
{
	if (image.getMipLevelCount() < 2)
	{
		return;
	}

	auto src_range  = image.getSubresourceRange();
	auto dst_range  = image.getSubresourceRange();
	auto src_layers = image.getSubresourceLayers();
	auto dst_layers = image.getSubresourceLayers();
	auto src_usage  = initial_usage;

	uint32_t src_width  = image.getWidth();
	uint32_t src_height = image.getHeight();
	uint32_t dst_width  = image.getWidth();
	uint32_t dst_height = image.getHeight();

	for (uint32_t i = 1; i < image.getMipLevelCount(); i++)
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

		std::array<VkImageMemoryBarrier, 2> barriers = {};
		// Transfer source
		barriers[0].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barriers[0].srcAccessMask       = Image::usage_to_access(src_usage);
		barriers[0].dstAccessMask       = Image::usage_to_access(VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
		barriers[0].oldLayout           = Image::usage_to_layout(src_usage);
		barriers[0].newLayout           = Image::usage_to_layout(VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
		barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[0].image               = image;
		barriers[0].subresourceRange    = src_range;
		// Transfer destination
		barriers[1].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barriers[1].srcAccessMask       = 0;
		barriers[1].dstAccessMask       = Image::usage_to_access(VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		barriers[1].oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
		barriers[1].newLayout           = Image::usage_to_layout(VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[1].image               = image;
		barriers[1].subresourceRange    = dst_range;

		vkCmdPipelineBarrier(*this, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());

		src_usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;

		VkImageBlit blit_info    = {};
		blit_info.srcOffsets[0]  = {0, 0, 0};
		blit_info.srcOffsets[1]  = {static_cast<int32_t>(src_width), static_cast<int32_t>(src_height), 1};
		blit_info.dstOffsets[0]  = {0, 0, 0};
		blit_info.dstOffsets[1]  = {static_cast<int32_t>(dst_width), static_cast<int32_t>(dst_height), 1};
		blit_info.srcSubresource = src_layers;
		blit_info.dstSubresource = dst_layers;

		vkCmdBlitImage(*this, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit_info, filter);
	}

	auto mip_level_range         = image.getSubresourceRange();
	mip_level_range.levelCount   = mip_level_range.levelCount - 1;
	VkImageMemoryBarrier barrier = {};
	barrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.srcAccessMask        = VK_ACCESS_TRANSFER_READ_BIT;
	barrier.dstAccessMask        = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.oldLayout            = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	barrier.newLayout            = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                = image;
	barrier.subresourceRange     = mip_level_range;

	vkCmdPipelineBarrier(*this, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void CommandBuffer::transferLayout(const Image &image, VkImageUsageFlagBits old_usage, VkImageUsageFlagBits new_usage) const
{
	auto                 subresource_range = image.getSubresourceRange();
	VkImageMemoryBarrier barrier           = {};
	barrier.sType                          = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.srcAccessMask                  = Image::usage_to_access(old_usage);
	barrier.dstAccessMask                  = Image::usage_to_access(new_usage);
	barrier.oldLayout                      = Image::usage_to_layout(old_usage);
	barrier.newLayout                      = Image::usage_to_layout(new_usage);
	barrier.srcQueueFamilyIndex            = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex            = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                          = image;
	barrier.subresourceRange               = subresource_range;

	vkCmdPipelineBarrier(*this, Image::usage_to_stage(old_usage), Image::usage_to_stage(new_usage), 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void CommandBuffer::transferLayout(const std::vector<ImageReference> &images, VkImageUsageFlagBits old_usage, VkImageUsageFlagBits new_usage) const
{
	std::vector<VkImageMemoryBarrier> barriers;

	for (auto &image : images)
	{
		auto subresource_range = image.get().getSubresourceRange();

		VkImageMemoryBarrier barrier = {};
		barrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.srcAccessMask        = Image::usage_to_access(old_usage);
		barrier.dstAccessMask        = Image::usage_to_access(new_usage);
		barrier.oldLayout            = Image::usage_to_layout(old_usage);
		barrier.newLayout            = Image::usage_to_layout(new_usage);
		barrier.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
		barrier.image                = image.get();
		barrier.subresourceRange     = subresource_range;

		barriers.push_back(barrier);
	}

	vkCmdPipelineBarrier(*this, Image::usage_to_stage(old_usage), Image::usage_to_stage(new_usage), 0, 0, nullptr, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());
}

void CommandBuffer::submitIdle(uint32_t index)
{
	std::lock_guard<std::mutex> lock(m_mutex);
	auto *                      queue = GraphicsContext::instance()->getQueueSystem().acquire(m_command_pool.getUsage(), index);
	queue->submitIdle(*this);
}

void CommandBuffer::submit(const VkSemaphore &wait_semaphore, const VkSemaphore &signal_semaphore, VkFence fence, VkShaderStageFlags wait_stages)
{
	std::lock_guard<std::mutex> lock(m_mutex);
	GraphicsContext::instance()->getQueueSystem().acquire(m_command_pool.getUsage())->submit(*this, signal_semaphore, wait_semaphore, fence, wait_stages);
}

void CommandBuffer::submit(const SubmitInfo &submit_info) const
{
	GraphicsContext::instance()->getQueueSystem().acquire(m_command_pool.getUsage())->submit(*this, submit_info);
}

CommandBuffer::operator const VkCommandBuffer &() const
{
	return m_handle;
}

const VkCommandBuffer &CommandBuffer::getCommandBuffer() const
{
	return m_handle;
}
}        // namespace Ilum