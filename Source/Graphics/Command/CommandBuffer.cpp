#include "CommandBuffer.hpp"
#include "CommandPool.hpp"
#include "Device/Device.hpp"

#include <array>

namespace Ilum::Graphics
{
CommandBuffer::CommandBuffer(const Device &device, const CommandPool &cmd_pool, VkCommandBufferLevel level) :
    m_device(device), m_pool(cmd_pool), m_level(level)
{
	VkCommandBufferAllocateInfo command_buffer_allocate_info = {};
	command_buffer_allocate_info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	command_buffer_allocate_info.commandPool                 = cmd_pool;
	command_buffer_allocate_info.level                       = level;
	command_buffer_allocate_info.commandBufferCount          = 1;
	vkAllocateCommandBuffers(m_device, &command_buffer_allocate_info, &m_handle);
}

CommandBuffer::~CommandBuffer()
{
	if (m_handle)
	{
		vkFreeCommandBuffers(m_device, m_pool, 1, &m_handle);
	}
}

CommandBuffer::operator const VkCommandBuffer &() const
{
	return m_handle;
}

const VkCommandBuffer &CommandBuffer::GetHandle() const
{
	return m_handle;
}

const CommandBuffer::State &CommandBuffer::GetState() const
{
	return m_state;
}

VkCommandBufferLevel CommandBuffer::GetLevel() const
{
	return m_level;
}

void CommandBuffer::Reset()
{
	if (m_pool.GetResetMode() == CommandPool::ResetMode::ResetIndividually)
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

void CommandBuffer::SubmitIdle(uint32_t queue_index)
{
	auto queue = m_device.GetQueue(m_pool.GetQueueFamily(), queue_index);
	vkQueueWaitIdle(queue);

	VkSubmitInfo submit_info       = {};
	submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &m_handle;

	VkFenceCreateInfo fence_create_info = {};
	fence_create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	VkFence fence                       = VK_NULL_HANDLE;
	vkCreateFence(m_device, &fence_create_info, nullptr, &fence);
	vkResetFences(m_device, 1, &fence);

	vkQueueSubmit(queue, 1, &submit_info, fence);

	vkWaitForFences(m_device, 1, &fence, true, std::numeric_limits<uint32_t>::max());
	vkDestroyFence(m_device, fence, nullptr);
}

void CommandBuffer::CopyImage(const ImageInfo& src, const ImageInfo& dst) const
{
	auto src_range = src.handle.get().GetSubresourceRange();
	auto dst_range = dst.handle.get().GetSubresourceRange();

	std::array<VkImageMemoryBarrier, 2> barriers = {};

	uint32_t barrier_count = 0;

	if (src.usage != VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
	{
		auto &src_barrier               = barriers[barrier_count++];
		src_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		src_barrier.srcAccessMask       = Graphics::Image::UsageToAccess(src.usage);
		src_barrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
		src_barrier.oldLayout           = Graphics::Image::UsageToLayout(src.usage);
		src_barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		src_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		src_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		src_barrier.image               = src.handle.get();
		src_barrier.subresourceRange    = src_range;
	}
	if (dst.usage != VK_IMAGE_USAGE_TRANSFER_DST_BIT)
	{
		auto &dst_barrier               = barriers[barrier_count++];
		dst_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		dst_barrier.srcAccessMask       = Graphics::Image::UsageToAccess(dst.usage);
		dst_barrier.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
		dst_barrier.oldLayout           = Graphics::Image::UsageToLayout(dst.usage);
		dst_barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		dst_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		dst_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		dst_barrier.image               = dst.handle.get();
		dst_barrier.subresourceRange    = dst_range;
	}

	if (barrier_count > 0)
	{
		vkCmdPipelineBarrier(*this, Graphics::Image::UsageToStage(src.usage) | Graphics::Image::UsageToStage(dst.usage), VK_PIPELINE_STAGE_TRANSFER_BIT, {}, 0, nullptr, 0, nullptr, barrier_count, barriers.data());
	}

	auto src_layers = src.handle.get().GetSubresourceLayers(src.mip_level, src.layer);
	auto dst_layers = dst.handle.get().GetSubresourceLayers(dst.mip_level, dst.layer);

	VkImageCopy copy_info    = {};
	copy_info.srcOffset      = {0, 0, 0};
	copy_info.dstOffset      = {0, 0, 0};
	copy_info.srcSubresource = src_layers;
	copy_info.dstSubresource = dst_layers;
	copy_info.extent         = {dst.handle.get().GetMipWidth(dst.mip_level),
                        dst.handle.get().GetMipHeight(dst.mip_level), 1};

	vkCmdCopyImage(*this, src.handle.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.handle.get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_info);
}

void CommandBuffer::CopyBuffer(const BufferInfo &src, const BufferInfo &dst, VkDeviceSize size) const
{
	ASSERT(src.handle.get().GetSize() >= src.offset + size);
	ASSERT(dst.handle.get().GetSize() >= dst.offset + size);

	VkBufferCopy copy_info = {};
	copy_info.dstOffset    = dst.offset;
	copy_info.size         = size;
	copy_info.srcOffset    = src.offset;

	vkCmdCopyBuffer(*this, src.handle.get(), dst.handle.get(), 1, &copy_info);
}

void CommandBuffer::CopyBufferToImage(const BufferInfo& src, const ImageInfo& dst) const
{
	if (dst.usage != VK_IMAGE_USAGE_TRANSFER_DST_BIT)
	{
		auto                 dst_range = dst.handle.get().GetSubresourceRange();
		VkImageMemoryBarrier barrier   = {};
		barrier.sType                  = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.srcAccessMask          = Graphics::Image::UsageToAccess(dst.usage);
		barrier.dstAccessMask          = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.oldLayout              = Graphics::Image::UsageToLayout(dst.usage);
		barrier.newLayout              = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.srcQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
		barrier.image                  = dst.handle.get();
		barrier.subresourceRange       = dst_range;

		vkCmdPipelineBarrier(*this, Graphics::Image::UsageToStage(dst.usage), VK_PIPELINE_STAGE_TRANSFER_BIT, {}, 0, nullptr, 0, nullptr, 1, &barrier);
	}

	auto dst_layer = dst.handle.get().GetSubresourceLayers(dst.mip_level, dst.layer);

	VkBufferImageCopy copy_info = {};
	copy_info.bufferOffset      = src.offset;
	copy_info.bufferImageHeight = 0;
	copy_info.bufferRowLength   = 0;
	copy_info.imageSubresource  = dst_layer;
	copy_info.imageOffset       = {0, 0, 0};
	copy_info.imageExtent       = {dst.handle.get().GetMipWidth(dst.mip_level),
                             dst.handle.get().GetMipHeight(dst.mip_level), 1};

	vkCmdCopyBufferToImage(*this, src.handle.get(), dst.handle.get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_info);
}

void CommandBuffer::CopyImageToBuffer(const ImageInfo &src, const BufferInfo &dst) const
{
	if (src.usage != VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
	{
		auto                 src_range = src.handle.get().GetSubresourceRange();
		VkImageMemoryBarrier barrier   = {};
		barrier.sType                  = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.srcAccessMask          = Graphics::Image::UsageToAccess(src.usage);
		barrier.dstAccessMask          = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.oldLayout              = Graphics::Image::UsageToLayout(src.usage);
		barrier.newLayout              = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.srcQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
		barrier.image                  = src.handle.get();
		barrier.subresourceRange       = src_range;

		vkCmdPipelineBarrier(*this, Graphics::Image::UsageToStage(src.usage), VK_PIPELINE_STAGE_TRANSFER_BIT, {}, 0, nullptr, 0, nullptr, 1, &barrier);
	}

	auto src_layer = src.handle.get().GetSubresourceLayers(src.mip_level, src.layer);

	VkBufferImageCopy copy_info = {};
	copy_info.bufferOffset      = dst.offset;
	copy_info.bufferImageHeight = src.handle.get().GetMipHeight(src.mip_level);
	copy_info.bufferRowLength   = src.handle.get().GetMipWidth(src.mip_level);
	copy_info.imageSubresource  = src_layer;
	copy_info.imageOffset       = {0, 0, 0};
	copy_info.imageExtent       = {src.handle.get().GetMipWidth(src.mip_level),
                             src.handle.get().GetMipHeight(src.mip_level), 1};

	vkCmdCopyImageToBuffer(*this, src.handle.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.handle.get(), 1, &copy_info);
}

void CommandBuffer::BlitImage(const Image &src, VkImageUsageFlagBits src_usage, const Image &dst, VkImageUsageFlagBits dst_usage, VkFilter filter) const
{
	auto src_range = src.GetSubresourceRange();
	auto dst_range = dst.GetSubresourceRange();

	std::array<VkImageMemoryBarrier, 2> barriers = {};

	uint32_t barrier_count = 0;

	if (src_usage != VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
	{
		auto &src_barrier               = barriers[barrier_count++];
		src_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		src_barrier.srcAccessMask       = Graphics::Image::UsageToAccess(src_usage);
		src_barrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
		src_barrier.oldLayout           = Graphics::Image::UsageToLayout(src_usage);
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
		dst_barrier.srcAccessMask       = Graphics::Image::UsageToAccess(dst_usage);
		dst_barrier.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
		dst_barrier.oldLayout           = Graphics::Image::UsageToLayout(dst_usage);
		dst_barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		dst_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		dst_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		dst_barrier.image               = dst;
		dst_barrier.subresourceRange    = dst_range;
	}

	if (barrier_count > 0)
	{
		vkCmdPipelineBarrier(*this, Graphics::Image::UsageToStage(src_usage) | Graphics::Image::UsageToStage(dst_usage), VK_PIPELINE_STAGE_TRANSFER_BIT, {}, 0, nullptr, 0, nullptr, barrier_count, barriers.data());
	}

	auto src_layers = src.GetSubresourceLayers();
	auto dst_layers = dst.GetSubresourceLayers();

	VkImageBlit blit_info    = {};
	blit_info.srcOffsets[0]  = {0, 0, 0};
	blit_info.srcOffsets[1]  = {static_cast<int32_t>(src.GetWidth()), static_cast<int32_t>(src.GetHeight()), 1};
	blit_info.dstOffsets[0]  = {0, 0, 0};
	blit_info.dstOffsets[1]  = {static_cast<int32_t>(src.GetWidth()), static_cast<int32_t>(src.GetHeight()), 1};
	blit_info.srcSubresource = src_layers;
	blit_info.dstSubresource = dst_layers;

	vkCmdBlitImage(*this, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit_info, filter);
}

void CommandBuffer::GenerateMipmap(const Image &image, VkImageUsageFlagBits initial_usage, VkFilter filter) const
{
	if (image.GetMipLevelCount() < 2)
	{
		return;
	}

	auto src_range  = image.GetSubresourceRange();
	auto dst_range  = image.GetSubresourceRange();
	auto src_layers = image.GetSubresourceLayers();
	auto dst_layers = image.GetSubresourceLayers();
	auto src_usage  = initial_usage;

	uint32_t src_width  = image.GetWidth();
	uint32_t src_height = image.GetHeight();
	uint32_t dst_width  = image.GetWidth();
	uint32_t dst_height = image.GetHeight();

	for (uint32_t i = 1; i < image.GetMipLevelCount(); i++)
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
		barriers[0].srcAccessMask       = Graphics::Image::UsageToAccess(src_usage);
		barriers[0].dstAccessMask       = Graphics::Image::UsageToAccess(VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
		barriers[0].oldLayout           = Graphics::Image::UsageToLayout(src_usage);
		barriers[0].newLayout           = Graphics::Image::UsageToLayout(VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
		barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[0].image               = image;
		barriers[0].subresourceRange    = src_range;
		// Transfer destination
		barriers[1].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barriers[1].srcAccessMask       = 0;
		barriers[1].dstAccessMask       = Graphics::Image::UsageToAccess(VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		barriers[1].oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
		barriers[1].newLayout           = Graphics::Image::UsageToLayout(VK_IMAGE_USAGE_TRANSFER_DST_BIT);
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

	auto mip_level_range         = image.GetSubresourceRange();
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

void CommandBuffer::TransferLayout(const Image &image, VkImageUsageFlagBits src_usage, VkImageUsageFlagBits dst_usage) const
{
	auto                 subresource_range = image.GetSubresourceRange();
	VkImageMemoryBarrier barrier           = {};
	barrier.sType                          = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.srcAccessMask                  = Graphics::Image::UsageToAccess(src_usage);
	barrier.dstAccessMask                  = Graphics::Image::UsageToAccess(dst_usage);
	barrier.oldLayout                      = Graphics::Image::UsageToLayout(src_usage);
	barrier.newLayout                      = Graphics::Image::UsageToLayout(dst_usage);
	barrier.srcQueueFamilyIndex            = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex            = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                          = image;
	barrier.subresourceRange               = subresource_range;

	vkCmdPipelineBarrier(*this, Graphics::Image::UsageToStage(src_usage), Graphics::Image::UsageToStage(dst_usage), 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void CommandBuffer::TransferLayout(const std::vector<ImageReference> &images, VkImageUsageFlagBits src_usage, VkImageUsageFlagBits dst_usage) const
{
	std::vector<VkImageMemoryBarrier> barriers;

	for (auto &image : images)
	{
		auto subresource_range = image.get().GetSubresourceRange();

		VkImageMemoryBarrier barrier = {};
		barrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.srcAccessMask        = Graphics::Image::UsageToAccess(src_usage);
		barrier.dstAccessMask        = Graphics::Image::UsageToAccess(dst_usage);
		barrier.oldLayout            = Graphics::Image::UsageToLayout(src_usage);
		barrier.newLayout            = Graphics::Image::UsageToLayout(dst_usage);
		barrier.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
		barrier.image                = image.get();
		barrier.subresourceRange     = subresource_range;
		barriers.push_back(barrier);
	}

	vkCmdPipelineBarrier(*this, Graphics::Image::UsageToStage(src_usage), Graphics::Image::UsageToStage(dst_usage), 0, 0, nullptr, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());
}
}        // namespace Ilum::Graphics