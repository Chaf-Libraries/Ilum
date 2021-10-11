#include "Image.hpp"

#include "Device/LogicalDevice.hpp"
#include "Device/PhysicalDevice.hpp"
#include "Device/Surface.hpp"
#include "Engine/Context.hpp"
#include "Engine/Engine.hpp"
#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/Command/CommandPool.hpp"
#include "Graphics/GraphicsContext.hpp"

namespace Ilum
{
static constexpr float ANISOTROPY = 16.f;

Image::Image(
    const VkExtent3D &    extent,
    VkFormat              format,
    VkImageUsageFlags     usage,
    uint32_t              mip_levels,
    uint32_t              array_layers,
    VkImageLayout         layout,
    VkFilter              filter,
    VkSamplerAddressMode  address_mode,
    VkSampleCountFlagBits samples) :
    m_extent(extent),
    m_format(format),
    m_usage(usage),
    m_mip_levels(mip_levels),
    m_array_layers(array_layers),
    m_layout(layout),
    m_filter(filter),
    m_address_mode(address_mode),
    m_samples(samples)
{
}

Image::~Image()
{
	auto graphics_context = Engine::instance()->getContext().getSubsystem<GraphicsContext>();

	if (m_view)
	{
		vkDestroyImageView(graphics_context->getLogicalDevice(), m_view, nullptr);
	}

	if (m_sampler)
	{
		vkDestroySampler(graphics_context->getLogicalDevice(), m_sampler, nullptr);
	}

	if (m_allocation)
	{
		vmaDestroyImage(graphics_context->getLogicalDevice().getAllocator(), m_image, m_allocation);
	}
}

const VkExtent3D &Image::getExtent() const
{
	return m_extent;
}

uint32_t Image::getWidth() const
{
	return m_extent.width;
}

uint32_t Image::getHeight() const
{
	return m_extent.height;
}

VkSampleCountFlagBits Image::getSamples() const
{
	return m_samples;
}

VkImageUsageFlags Image::getUsage() const
{
	return m_usage;
}

VkFormat Image::getFormat() const
{
	return m_format;
}

uint32_t Image::getMipLevels() const
{
	return m_mip_levels;
}

uint32_t Image::getArrayLevels() const
{
	return m_array_layers;
}

VkFilter Image::getFilter() const
{
	return m_filter;
}

VkSamplerAddressMode Image::getAddressMode() const
{
	return m_address_mode;
}

VkImageLayout Image::getImageLayout() const
{
	return m_layout;
}

const VkImage &Image::getImage() const
{
	return m_image;
}

const VkImageView &Image::getView() const
{
	return m_view;
}

const VkSampler &Image::getSampler() const
{
	return m_sampler;
}

const VkDescriptorImageInfo& Image::getDescriptor() const
{
	return m_descriptor;
}

uint32_t Image::getMipLevels(const VkExtent3D &extent)
{
	return static_cast<uint32_t>(std::floorf(std::log2f(std::fmaxf(static_cast<float>(extent.width), std::fmaxf(static_cast<float>(extent.height), static_cast<float>(extent.depth))))) + 1);
}

bool Image::hasDepth(VkFormat format)
{
	const std::vector<VkFormat> depth_formats = {
	    VK_FORMAT_D16_UNORM,
	    VK_FORMAT_X8_D24_UNORM_PACK32,
	    VK_FORMAT_D32_SFLOAT,
	    VK_FORMAT_D16_UNORM_S8_UINT,
	    VK_FORMAT_D24_UNORM_S8_UINT,
	    VK_FORMAT_D32_SFLOAT_S8_UINT};

	return std::find(depth_formats.begin(), depth_formats.end(), format) != depth_formats.end();
}

bool Image::hasStencil(VkFormat format)
{
	const std::vector<VkFormat> stencil_formats = {
	    VK_FORMAT_S8_UINT,
	    VK_FORMAT_D16_UNORM_S8_UINT,
	    VK_FORMAT_D24_UNORM_S8_UINT,
	    VK_FORMAT_D32_SFLOAT_S8_UINT};

	return std::find(stencil_formats.begin(), stencil_formats.end(), format) != stencil_formats.end();
}

VkFormat Image::findSupportedFormat(const std::vector<VkFormat> &formats, VkImageTiling tiling, VkFormatFeatureFlags features)
{
	for (const auto &format : formats)
	{
		VkFormatProperties properties;
		vkGetPhysicalDeviceFormatProperties(Engine::instance()->getContext().getSubsystem<GraphicsContext>()->getPhysicalDevice(), format, &properties);

		if (tiling == VK_IMAGE_TILING_LINEAR && (properties.linearTilingFeatures & features) == features)
		{
			return format;
		}
		if (tiling == VK_IMAGE_TILING_OPTIMAL && (properties.optimalTilingFeatures & features) == features)
		{
			return format;
		}
	}

	return VK_FORMAT_UNDEFINED;
}

bool Image::createImage(
    VkImage &             image,
    VmaAllocation &       allocation,
    const VkExtent3D &    extent,
    VkImageType           type,
    VkFormat              format,
    uint32_t              mip_levels,
    uint32_t              array_layers,
    VkSampleCountFlagBits samples,
    VkImageTiling         tiling,
    VkImageUsageFlags     usage,
    VmaMemoryUsage        memory_usage)
{
	VkImageCreateInfo image_create_info = {};
	image_create_info.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_create_info.flags             = array_layers == 6 ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
	image_create_info.imageType         = type;
	image_create_info.format            = format;
	image_create_info.extent            = extent;
	image_create_info.mipLevels         = mip_levels;
	image_create_info.arrayLayers       = array_layers;
	image_create_info.samples           = samples;
	image_create_info.tiling            = tiling;
	image_create_info.usage             = usage;
	image_create_info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
	image_create_info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;

	VmaAllocationCreateInfo allocation_create_info = {};
	allocation_create_info.usage                   = memory_usage;

	if (!VK_CHECK(vmaCreateImage(Engine::instance()->getContext().getSubsystem<GraphicsContext>()->getLogicalDevice().getAllocator(), &image_create_info, &allocation_create_info, &image, &allocation, nullptr)))
	{
		VK_ERROR("Failed to create image!");
		return false;
	}

	return true;
}

bool Image::createImageView(
    const VkImage &    image,
    VkImageView &      image_view,
    VkImageViewType    type,
    VkFormat           format,
    uint32_t           mip_levels,
    uint32_t           base_mip_level,
    uint32_t           layer_count,
    uint32_t           base_array_layer,
    VkImageAspectFlags image_aspect)
{
	VkImageViewCreateInfo image_view_create_info           = {};
	image_view_create_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	image_view_create_info.image                           = image;
	image_view_create_info.viewType                        = type;
	image_view_create_info.format                          = format;
	image_view_create_info.components                      = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
	image_view_create_info.subresourceRange.aspectMask     = image_aspect;
	image_view_create_info.subresourceRange.baseMipLevel   = base_mip_level;
	image_view_create_info.subresourceRange.levelCount     = mip_levels;
	image_view_create_info.subresourceRange.baseArrayLayer = base_array_layer;
	image_view_create_info.subresourceRange.layerCount     = layer_count;

	if (!VK_CHECK(vkCreateImageView(Engine::instance()->getContext().getSubsystem<GraphicsContext>()->getLogicalDevice(), &image_view_create_info, nullptr, &image_view)))
	{
		VK_ERROR("Failed to create image view!");
		return false;
	}

	return true;
}

bool Image::createImageSampler(
    VkSampler &          sampler,
    VkFilter             filter,
    VkSamplerAddressMode address_mode,
    bool                 anisotropic,
    uint32_t             mip_levels)
{
	auto graphics_context = Engine::instance()->getContext().getSubsystem<GraphicsContext>();

	VkSamplerCreateInfo sampler_create_info = {};
	sampler_create_info.sType               = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler_create_info.magFilter           = filter;
	sampler_create_info.minFilter           = filter;
	sampler_create_info.mipmapMode          = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	sampler_create_info.addressModeU        = address_mode;
	sampler_create_info.addressModeV        = address_mode;
	sampler_create_info.addressModeW        = address_mode;
	sampler_create_info.mipLodBias          = 0.f;
	sampler_create_info.maxAnisotropy =
	    (anisotropic && graphics_context->getLogicalDevice().getEnabledFeatures().samplerAnisotropy) ? std::min(ANISOTROPY, graphics_context->getPhysicalDevice().getProperties().limits.maxSamplerAnisotropy) : 1.f;
	sampler_create_info.compareEnable           = VK_FALSE;
	sampler_create_info.compareOp               = VK_COMPARE_OP_ALWAYS;
	sampler_create_info.minLod                  = 0.f;
	sampler_create_info.maxLod                  = static_cast<float>(mip_levels);
	sampler_create_info.borderColor             = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	sampler_create_info.unnormalizedCoordinates = VK_FALSE;

	if (!VK_CHECK(vkCreateSampler(graphics_context->getLogicalDevice(), &sampler_create_info, nullptr, &sampler)))
	{
		VK_ERROR("Failed to create sampler!");
		return false;
	}

	return true;
}

void Image::createMipmaps(
    const VkImage &   image,
    const VkExtent3D &extent,
    VkFormat          format,
    VkImageLayout     dst_image_layout,
    uint32_t          mip_levels,
    uint32_t          base_array_layer,
    uint32_t          layer_count)
{
	VkFormatProperties format_properties;
	vkGetPhysicalDeviceFormatProperties(Engine::instance()->getContext().getSubsystem<GraphicsContext>()->getPhysicalDevice(), format, &format_properties);

	// Check blit supporting
	ASSERT(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT);
	ASSERT(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);

	CommandBuffer command_buffer;
	command_buffer.begin();

	/*
		vkb::insert_image_memory_barrier(
	    copy_command,
	    texture.image,
	    VK_ACCESS_TRANSFER_WRITE_BIT,
	    VK_ACCESS_TRANSFER_READ_BIT,
	    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	    VK_PIPELINE_STAGE_TRANSFER_BIT,
	    VK_PIPELINE_STAGE_TRANSFER_BIT,
	    {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
	*/

	for (uint32_t i = 1; i < mip_levels; i++)
	{
		VkImageBlit image_blit = {};

		// Source
		image_blit.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		image_blit.srcSubresource.mipLevel       = i - 1;
		image_blit.srcSubresource.baseArrayLayer = base_array_layer;
		image_blit.srcSubresource.layerCount     = layer_count;
		image_blit.srcOffsets[1].x               = int32_t(extent.width >> (i - 1));
		image_blit.srcOffsets[1].y               = int32_t(extent.height >> (i - 1));
		image_blit.srcOffsets[1].z               = 1;

		// Destination
		image_blit.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		image_blit.dstSubresource.mipLevel       = i;
		image_blit.dstSubresource.baseArrayLayer = base_array_layer;
		image_blit.dstSubresource.layerCount     = layer_count;
		image_blit.dstOffsets[1].x               = int32_t(extent.width >> i);
		image_blit.dstOffsets[1].y               = int32_t(extent.height >> i);
		image_blit.dstOffsets[1].z               = 1;

		// TODO: Sync issue
		insertImageMemoryBarrier(
		    command_buffer,
		    image,
		    VK_ACCESS_TRANSFER_WRITE_BIT,
		    VK_ACCESS_TRANSFER_READ_BIT,
		    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		    VK_PIPELINE_STAGE_TRANSFER_BIT,
		    VK_PIPELINE_STAGE_TRANSFER_BIT,
		    VK_IMAGE_ASPECT_COLOR_BIT,
		    1,
		    i - 1,
		    layer_count,
		    base_array_layer);

		vkCmdBlitImage(
		    command_buffer,
		    image,
		    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		    image,
		    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		    1,
		    &image_blit,
		    VK_FILTER_LINEAR);

		insertImageMemoryBarrier(
		    command_buffer,
		    image,
		    VK_ACCESS_TRANSFER_READ_BIT,
		    VK_ACCESS_SHADER_READ_BIT,
		    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		    dst_image_layout,
		    VK_PIPELINE_STAGE_TRANSFER_BIT,
		    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		    VK_IMAGE_ASPECT_COLOR_BIT,
		    1,
		    i - 1,
		    layer_count,
		    base_array_layer);
	}

	insertImageMemoryBarrier(
	    command_buffer,
	    image,
	    VK_ACCESS_TRANSFER_WRITE_BIT,
	    VK_ACCESS_SHADER_READ_BIT,
	    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	    dst_image_layout,
	    VK_PIPELINE_STAGE_TRANSFER_BIT,
	    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
	    VK_IMAGE_ASPECT_COLOR_BIT,
	    1,
	    mip_levels - 1,
	    layer_count,
	    base_array_layer);

	command_buffer.end();
	command_buffer.submitIdle();
}

void Image::transitionImageLayout(
    const VkImage &    image,
    VkFormat           format,
    VkImageLayout      src_image_layout,
    VkImageLayout      dst_image_layout,
    VkImageAspectFlags image_aspect,
    uint32_t           mip_levels,
    uint32_t           base_mip_level,
    uint32_t           layer_count,
    uint32_t           base_array_layer)
{
	CommandBuffer command_buffer;
	command_buffer.begin();

	VkAccessFlags src_access_mask = 0;
	VkAccessFlags dst_access_mask = 0;

	switch (src_image_layout)
	{
		case VK_IMAGE_LAYOUT_UNDEFINED:
			src_access_mask = 0;
			break;
		case VK_IMAGE_LAYOUT_PREINITIALIZED:
			src_access_mask = VK_ACCESS_HOST_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
			src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
			src_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
			src_access_mask = VK_ACCESS_TRANSFER_READ_BIT;
			break;
		case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
			src_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
			src_access_mask = VK_ACCESS_SHADER_READ_BIT;
			break;
		default:
			break;
	}

	switch (dst_image_layout)
	{
		case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
			dst_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
			dst_access_mask = VK_ACCESS_TRANSFER_READ_BIT;
			break;
		case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
			dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
			dst_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
			if (src_access_mask == 0)
			{
				src_access_mask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
			}
			dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
			break;
		default:
			break;
	}

	insertImageMemoryBarrier(
	    command_buffer,
	    image,
	    src_access_mask,
	    dst_access_mask,
	    src_image_layout,
	    dst_image_layout,
	    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
	    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
	    image_aspect,
	    mip_levels,
	    base_mip_level,
	    layer_count,
	    base_array_layer);

	command_buffer.end();
	command_buffer.submitIdle();
}

void Image::insertImageMemoryBarrier(
    const CommandBuffer &command_buffer,
    const VkImage &      image,
    VkAccessFlags        src_access_mask,
    VkAccessFlags        dst_access_mask,
    VkImageLayout        old_image_layout,
    VkImageLayout        new_image_layout,
    VkPipelineStageFlags src_stage_mask,
    VkPipelineStageFlags dst_stage_mask,
    VkImageAspectFlags   image_aspect,
    uint32_t             mip_levels,
    uint32_t             base_mip_level,
    uint32_t             layer_count,
    uint32_t             base_array_layer)
{
	VkImageMemoryBarrier barrier            = {};
	barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.srcAccessMask                   = src_access_mask;
	barrier.dstAccessMask                   = dst_access_mask;
	barrier.oldLayout                       = old_image_layout;
	barrier.newLayout                       = new_image_layout;
	barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                           = image;
	barrier.subresourceRange.aspectMask     = image_aspect;
	barrier.subresourceRange.baseMipLevel   = base_mip_level;
	barrier.subresourceRange.levelCount     = mip_levels;
	barrier.subresourceRange.baseArrayLayer = base_array_layer;
	barrier.subresourceRange.layerCount     = layer_count;
	vkCmdPipelineBarrier(command_buffer, src_stage_mask, dst_stage_mask, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void Image::copyBufferToImage(
    const VkBuffer &  buffer,
    const VkImage &   image,
    const VkExtent3D &extent,
    uint32_t          layer_count,
    uint32_t          base_array_layer)
{
	CommandBuffer command_buffer;
	command_buffer.begin();

	VkBufferImageCopy region               = {};
	region.bufferOffset                    = 0;
	region.bufferRowLength                 = 0;
	region.bufferImageHeight               = 0;
	region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel       = 0;
	region.imageSubresource.baseArrayLayer = base_array_layer;
	region.imageSubresource.layerCount     = layer_count;
	region.imageOffset                     = {0, 0, 0};
	region.imageExtent                     = extent;
	vkCmdCopyBufferToImage(command_buffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	command_buffer.end();
	command_buffer.submitIdle();
}

void Image::copyImage(
    const VkImage &   src_image,
    VkImage &         dst_image,
    VmaAllocation &   dst_image_allocation,
    VkFormat          src_format,
    const VkExtent3D &extent,
    VkImageLayout     src_image_layout,
    uint32_t          mip_level,
    uint32_t          array_layer)
{
	auto &physical_device = Engine::instance()->getContext().getSubsystem<GraphicsContext>()->getPhysicalDevice();
	auto &surface         = Engine::instance()->getContext().getSubsystem<GraphicsContext>()->getSurface();

	CommandBuffer command_buffer;
	command_buffer.begin();

	auto               support_blit = true;
	VkFormatProperties format_properties;

	vkGetPhysicalDeviceFormatProperties(physical_device, surface.getFormat().format, &format_properties);

	if (!(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT))
	{
		VK_WARN("Device doesn't support blitting from optimal tiled images, using copy instead of blit!");
		support_blit = false;
	}

	vkGetPhysicalDeviceFormatProperties(physical_device, src_format, &format_properties);

	if (!(format_properties.linearTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT))
	{
		VK_WARN("Device doesn't support blitting to linear tiled images, using copy instead of blit!");
		support_blit = false;
	}

	createImage(dst_image, dst_image_allocation, extent, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_LINEAR,
	            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

	if (support_blit)
	{
		VkOffset3D blit_size = {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), static_cast<int32_t>(extent.depth)};

		VkImageBlit blit_region                   = {};
		blit_region.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		blit_region.srcSubresource.mipLevel       = mip_level;
		blit_region.srcSubresource.baseArrayLayer = array_layer;
		blit_region.srcSubresource.layerCount     = 1;
		blit_region.srcOffsets[1]                 = blit_size;
		blit_region.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		blit_region.dstSubresource.mipLevel       = 0;
		blit_region.dstSubresource.baseArrayLayer = 0;
		blit_region.dstSubresource.layerCount     = 1;
		blit_region.dstOffsets[1]                 = blit_size;
		vkCmdBlitImage(command_buffer, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit_region, VK_FILTER_NEAREST);
	}
	else
	{
		VkImageCopy copy_region                   = {};
		copy_region.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		copy_region.srcSubresource.mipLevel       = mip_level;
		copy_region.srcSubresource.baseArrayLayer = array_layer;
		copy_region.srcSubresource.layerCount     = 1;
		copy_region.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		copy_region.dstSubresource.mipLevel       = 0;
		copy_region.dstSubresource.baseArrayLayer = 0;
		copy_region.dstSubresource.layerCount     = 1;
		copy_region.extent                        = extent;
		vkCmdCopyImage(command_buffer, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);
	}

	insertImageMemoryBarrier(command_buffer, dst_image, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
	                         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1, 0);

	insertImageMemoryBarrier(command_buffer, src_image, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, src_image_layout,
	                         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 1, mip_level, 1, array_layer);

	command_buffer.end();
	command_buffer.submitIdle();
}

void Image::updateDescriptor()
{
	m_descriptor.imageLayout = m_layout;
	m_descriptor.imageView   = m_view;
	m_descriptor.sampler     = m_sampler;
}
}        // namespace Ilum