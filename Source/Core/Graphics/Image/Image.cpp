#include "Image.hpp"

#include "Core/Device/LogicalDevice.hpp"
#include "Core/Device/PhysicalDevice.hpp"
#include "Core/Graphics/Command/CommandBuffer.hpp"
#include "Core/Graphics/Command/CommandPool.hpp"

namespace Ilum
{
static constexpr float ANISOTROPY = 16.f;

Image::Image(
    const LogicalDevice & logical_device,
    const VkExtent3D &    extent,
    VkFormat              format,
    VkImageUsageFlags     usage,
    uint32_t              mip_levels,
    uint32_t              array_layers,
    VkImageLayout         layout,
    VkFilter              filter,
    VkSamplerAddressMode  address_mode,
    VkSampleCountFlagBits samples) :
    m_logical_device(logical_device),
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
	if (m_view)
	{
		vkDestroyImageView(m_logical_device, m_view, nullptr);
	}

	if (m_sampler)
	{
		vkDestroySampler(m_logical_device, m_sampler, nullptr);
	}

	if (m_allocation)
	{
		vmaDestroyImage(m_logical_device.getAllocator(), m_image, m_allocation);
	}

	if (m_image)
	{
		vkDestroyImage(m_logical_device, m_image, nullptr);
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

uint32_t Image::getMipLevels(const VkExtent3D &extent)
{
	return static_cast<uint32_t>(std::floorf(std::log2f(std::fmaxf(extent.width, std::fmaxf(extent.height, extent.depth)))) + 1);
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

VkFormat Image::findSupportedFormat(const LogicalDevice &logical_device, const std::vector<VkFormat> &formats, VkImageTiling tiling, VkFormatFeatureFlags features)
{
	for (const auto &format : formats)
	{
		VkFormatProperties properties;
		vkGetPhysicalDeviceFormatProperties(logical_device.getPhysicalDevice(), format, &properties);

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
    const LogicalDevice & logical_device,
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

	if (!VK_CHECK(vmaCreateImage(logical_device.getAllocator(), &image_create_info, &allocation_create_info, &image, &allocation, nullptr)))
	{
		VK_ERROR("Failed to create image!");
		return false;
	}

	return true;
}

bool Image::createImageView(
    const LogicalDevice &logical_device,
    const VkImage &      image,
    VkImageView &        image_view,
    VkImageViewType      type,
    VkFormat             format,
    uint32_t             mip_levels,
    uint32_t             base_mip_level,
    uint32_t             layer_count,
    uint32_t             base_array_layer,
    VkImageAspectFlags   image_aspect)
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

	if (!VK_CHECK(vkCreateImageView(logical_device, &image_view_create_info, nullptr, &image_view)))
	{
		VK_ERROR("Failed to create image view!");
		return false;
	}

	return true;
}

bool Image::createImageSampler(
    const LogicalDevice &logical_device,
    VkSampler &          sampler,
    VkFilter             filter,
    VkSamplerAddressMode address_mode,
    bool                 anisotropic,
    uint32_t             mip_levels)
{
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
	    (anisotropic && logical_device.getEnabledFeatures().samplerAnisotropy) ? std::min(ANISOTROPY, logical_device.getPhysicalDevice().getProperties().limits.maxSamplerAnisotropy) : 1.f;
	sampler_create_info.compareEnable           = VK_FALSE;
	sampler_create_info.compareOp               = VK_COMPARE_OP_ALWAYS;
	sampler_create_info.minLod                  = 0.f;
	sampler_create_info.maxLod                  = static_cast<float>(mip_levels);
	sampler_create_info.borderColor             = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	sampler_create_info.unnormalizedCoordinates = VK_FALSE;

	if (!VK_CHECK(vkCreateSampler(logical_device, &sampler_create_info, nullptr, &sampler)))
	{
		VK_ERROR("Failed to create sampler!");
		return false;
	}

	return true;
}

void Image::createMipmaps(
    const CommandPool &command_pool,
    const VkImage &    image,
    const VkExtent3D & extent,
    VkFormat           format,
    VkImageLayout      dst_image_layout,
    uint32_t           mip_levels,
    uint32_t           base_array_layer,
    uint32_t           layer_count)
{
	VkFormatProperties format_properties;
	vkGetPhysicalDeviceFormatProperties(command_pool.getLogicalDevice().getPhysicalDevice(), format, &format_properties);

	// Check blit supporting
	ASSERT(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT);
	ASSERT(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);

	CommandBuffer command_buffer(command_pool);
	command_buffer.begin();

	VkImageMemoryBarrier barrier = {};

	for (uint32_t i = 1; i < mip_levels; i++)
	{
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

		VkImageBlit image_blit                   = {};
		image_blit.srcOffsets[1]                 = {int32_t(extent.width >> (i - 1)), int32_t(extent.height >> (i - 1)), 1};
		image_blit.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		image_blit.srcSubresource.mipLevel       = i - 1;
		image_blit.srcSubresource.baseArrayLayer = base_array_layer;
		image_blit.srcSubresource.layerCount     = layer_count;
		image_blit.dstOffsets[1]                 = {int32_t(extent.width >> i), int32_t(extent.height >> i), 1};
		image_blit.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		image_blit.dstSubresource.mipLevel       = i;
		image_blit.dstSubresource.baseArrayLayer = base_array_layer;
		image_blit.dstSubresource.layerCount     = layer_count;
		vkCmdBlitImage(command_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &image_blit, VK_FILTER_LINEAR);

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
    const CommandPool &command_pool,
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
	CommandBuffer command_buffer(command_pool);
	command_buffer.begin();

	VkImageMemoryBarrier barrier            = {};
	barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout                       = src_image_layout;
	barrier.newLayout                       = dst_image_layout;
	barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                           = image;
	barrier.subresourceRange.aspectMask     = image_aspect;
	barrier.subresourceRange.baseMipLevel   = base_mip_level;
	barrier.subresourceRange.levelCount     = mip_levels;
	barrier.subresourceRange.baseArrayLayer = base_array_layer;
	barrier.subresourceRange.layerCount     = layer_count;

	switch (src_image_layout)
	{
		case VK_IMAGE_LAYOUT_UNDEFINED:
			barrier.srcAccessMask = 0;
			break;
		case VK_IMAGE_LAYOUT_PREINITIALIZED:
			barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
			barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
			barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			break;
		case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
			barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
			break;
		default:
			break;
	}

	switch (dst_image_layout)
	{
		case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			break;
		case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
			barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
			barrier.dstAccessMask = barrier.dstAccessMask | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			break;
		case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
			if (barrier.srcAccessMask == 0)
			{
				barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
			}
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			break;
		default:
			break;
	}

	vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

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
}        // namespace Ilum