#include "Image.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Device/LogicalDevice.hpp"

namespace Ilum
{
inline VkImageViewType get_image_view_type(const Image &image)
{
	if (image.getLayerCount() == 1)
	{
		return VK_IMAGE_VIEW_TYPE_2D;
	}
	else if (image.getLayerCount() == 6)
	{
		return VK_IMAGE_VIEW_TYPE_CUBE;
	}
	else
	{
		return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
	}
}

VkImageAspectFlags Image::format_to_aspect(VkFormat format)
{
	switch (format)
	{
		case VK_FORMAT_D16_UNORM:
		case VK_FORMAT_X8_D24_UNORM_PACK32:
		case VK_FORMAT_D32_SFLOAT:
			return VK_IMAGE_ASPECT_DEPTH_BIT;
		case VK_FORMAT_D16_UNORM_S8_UINT:
		case VK_FORMAT_D24_UNORM_S8_UINT:
		case VK_FORMAT_D32_SFLOAT_S8_UINT:
			return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
		default:
			return VK_IMAGE_ASPECT_COLOR_BIT;
	}
}

VkImageLayout Image::usage_to_layout(VkImageUsageFlagBits usage)
{
	switch (usage)
	{
		case VK_IMAGE_USAGE_TRANSFER_SRC_BIT:
			return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		case VK_IMAGE_USAGE_TRANSFER_DST_BIT:
			return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		case VK_IMAGE_USAGE_SAMPLED_BIT:
			return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		case VK_IMAGE_USAGE_STORAGE_BIT:
			return VK_IMAGE_LAYOUT_GENERAL;
		case VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT:
			return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		case VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT:
			return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		case VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT:
			return VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
		case VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR:
			return VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;
		default:
			return VK_IMAGE_LAYOUT_UNDEFINED;
	}
}

VkAccessFlags Image::usage_to_access(VkImageUsageFlagBits usage)
{
	switch (usage)
	{
		case VK_IMAGE_USAGE_TRANSFER_SRC_BIT:
			return VK_ACCESS_TRANSFER_READ_BIT;
		case VK_IMAGE_USAGE_TRANSFER_DST_BIT:
			return VK_ACCESS_TRANSFER_WRITE_BIT;
		case VK_IMAGE_USAGE_SAMPLED_BIT:
			return VK_ACCESS_SHADER_READ_BIT;
		case VK_IMAGE_USAGE_STORAGE_BIT:
			return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		case VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT:
			return VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		case VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT:
			return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		case VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT:
			return VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
		case VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR:
			return VK_ACCESS_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR;
		default:
			return VK_ACCESS_NONE_KHR;
	}
}

VkPipelineStageFlags Image::usage_to_stage(VkImageUsageFlagBits usage)
{
	switch (usage)
	{
		case VK_IMAGE_USAGE_TRANSFER_SRC_BIT:
			return VK_PIPELINE_STAGE_TRANSFER_BIT;
		case VK_IMAGE_USAGE_TRANSFER_DST_BIT:
			return VK_PIPELINE_STAGE_TRANSFER_BIT;
		case VK_IMAGE_USAGE_SAMPLED_BIT:
			return VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
		case VK_IMAGE_USAGE_STORAGE_BIT:
			return VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
		case VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT:
			return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		case VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT:
			return VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		case VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT:
			return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		case VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR:
			return VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
		default:
			return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	}
}

Image::Image(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VmaMemoryUsage memory_usage, bool mip_level, uint32_t layer_count) :
    m_extent({width, height}), m_format(format)
{
	// Calculate mip levels
	m_mip_level_count = mip_level ? static_cast<uint32_t>(std::floor(std::log2(std::max(width, height))) + 1) : 1;
	m_layer_count     = layer_count;

	// Create VkImage
	VkImageCreateInfo image_create_info = {};
	image_create_info.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_create_info.imageType         = VK_IMAGE_TYPE_2D;
	image_create_info.format            = m_format;
	image_create_info.extent            = VkExtent3D{width, height, 1};
	image_create_info.samples           = VK_SAMPLE_COUNT_1_BIT;
	image_create_info.mipLevels         = m_mip_level_count;
	image_create_info.arrayLayers       = m_layer_count;
	image_create_info.tiling            = VK_IMAGE_TILING_OPTIMAL;
	image_create_info.usage             = usage;
	image_create_info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
	image_create_info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;

	if (layer_count == 6)
	{
		image_create_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	}

	// Allocation memory for image
	VmaAllocationCreateInfo allocation_create_info = {};
	allocation_create_info.usage                   = memory_usage;
	vmaCreateImage(GraphicsContext::instance()->getLogicalDevice().getAllocator(), &image_create_info, &allocation_create_info, &m_handle, &m_allocation, nullptr);

	createImageViews();
}

Image::Image(VkImage image, uint32_t width, uint32_t height, VkFormat format) :
    m_handle(image), m_extent({width, height}), m_format(format)
{
	createImageViews();
}

Image::Image(Image &&other) noexcept :
    m_handle(other.m_handle),
    m_views(std::move(other.m_views)),
    m_layer_views(std ::move(other.m_layer_views)),
    m_extent(other.m_extent),
    m_mip_level_count(other.m_mip_level_count),
    m_layer_count(other.m_layer_count),
    m_format(other.m_format),
    m_allocation(other.m_allocation)
{
	other.m_handle     = VK_NULL_HANDLE;
	other.m_allocation = VK_NULL_HANDLE;
}

Image &Image::operator=(Image &&other) noexcept
{
	destroy();

	m_handle          = other.m_handle;
	m_views           = std::move(other.m_views);
	m_layer_views     = std ::move(other.m_layer_views);
	m_extent          = other.m_extent;
	m_mip_level_count = other.m_mip_level_count;
	m_layer_count     = other.m_layer_count;
	m_format          = other.m_format;
	m_allocation      = other.m_allocation;

	other.m_handle     = VK_NULL_HANDLE;
	other.m_allocation = VK_NULL_HANDLE;

	return *this;
}

Image::~Image()
{
	destroy();
}

const VkImageView &Image::getView(ImageViewType type) const
{
	switch (type)
	{
		case Ilum::ImageViewType::Native:
			return m_views.native;
		case Ilum::ImageViewType::Depth_Only:
			return m_views.depth;
		case Ilum::ImageViewType::Stencil_Only:
			return m_views.stencil;
		default:
			break;
	}

	return m_views.native;
}

const VkImageView &Image::getView(uint32_t layer, ImageViewType type) const
{
	if (m_layer_count == 1)
	{
		return getView(type);
	}

	switch (type)
	{
		case Ilum::ImageViewType::Native:
			return m_layer_views[layer].native;
		case Ilum::ImageViewType::Depth_Only:
			return m_layer_views[layer].depth;
		case Ilum::ImageViewType::Stencil_Only:
			return m_layer_views[layer].stencil;
		default:
			break;
	}
	return m_layer_views[layer].native;
}

uint32_t Image::getWidth() const
{
	return m_extent.width;
}

uint32_t Image::getHeight() const
{
	return m_extent.height;
}

uint32_t Image::getMipWidth(uint32_t mip_level) const
{
	return std::max(m_extent.width, 1u << mip_level) >> mip_level;
}

uint32_t Image::getMipHeight(uint32_t mip_level) const
{
	return std::max(m_extent.height, 1u << mip_level) >> mip_level;
}

uint32_t Image::getMipLevelCount() const
{
	return m_mip_level_count;
}
uint32_t Image::getLayerCount() const
{
	return m_layer_count;
}

VkFormat Image::getFormat() const
{
	return m_format;
}

const VkImage &Image::getImage() const
{
	return m_handle;
}

Image::operator const VkImage &() const
{
	return m_handle;
}

VkImageSubresourceLayers Image::getSubresourceLayers(uint32_t mip_level, uint32_t layer) const
{
	return VkImageSubresourceLayers{
	    /*aspectMask*/ format_to_aspect(m_format),
	    /*mipLevel*/ mip_level,
	    /*baseArrayLayer*/ layer,
	    /*layerCount*/ 1};
}

VkImageSubresourceRange Image::getSubresourceRange() const
{
	return VkImageSubresourceRange{
	    /*aspectMask*/ format_to_aspect(m_format),
	    /*baseMipLevel*/ 0,
	    /*levelCount*/ m_mip_level_count,
	    /*baseArrayLayer*/ 0,
	    /*layerCount*/ m_layer_count};
}

bool Image::isDepth() const
{
	return m_format == VK_FORMAT_D32_SFLOAT || m_format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

bool Image::isStencil() const
{
	return m_format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

void Image::createImageViews()
{
	auto subresource_range = getSubresourceRange();

	VkImageViewCreateInfo view_create_info = {};
	view_create_info.sType                 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	view_create_info.image                 = m_handle;
	view_create_info.viewType              = get_image_view_type(*this);
	view_create_info.format                = m_format;
	view_create_info.components            = {
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY};
	view_create_info.subresourceRange = subresource_range;

	// Create default image view
	// Native
	auto native_range                 = getSubresourceRange();
	view_create_info.subresourceRange = native_range;
	vkCreateImageView(GraphicsContext::instance()->getLogicalDevice(), &view_create_info, nullptr, &m_views.native);

	// Depth
	auto depth_range = getSubresourceRange();
	depth_range.aspectMask &= VK_IMAGE_ASPECT_DEPTH_BIT;
	if (depth_range.aspectMask != 0)
	{
		view_create_info.subresourceRange = depth_range;
		vkCreateImageView(GraphicsContext::instance()->getLogicalDevice(), &view_create_info, nullptr, &m_views.depth);
	}

	// Stencil
	auto stencil_range = getSubresourceRange();
	stencil_range.aspectMask &= VK_IMAGE_ASPECT_STENCIL_BIT;
	if (stencil_range.aspectMask != 0)
	{
		view_create_info.subresourceRange = stencil_range;
		vkCreateImageView(GraphicsContext::instance()->getLogicalDevice(), &view_create_info, nullptr, &m_views.stencil);
	}

	// Create layer sub image views
	if (m_layer_count > 1)
	{
		m_layer_views.resize(m_layer_count);
		view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
	}

	uint32_t layer = 0;
	for (auto &views : m_layer_views)
	{
		// Native
		native_range                      = getSubresourceRange();
		native_range.baseArrayLayer       = layer;
		native_range.layerCount           = 1;
		view_create_info.subresourceRange = native_range;
		vkCreateImageView(GraphicsContext::instance()->getLogicalDevice(), &view_create_info, nullptr, &views.native);

		// Depth
		depth_range = getSubresourceRange();
		depth_range.aspectMask &= VK_IMAGE_ASPECT_DEPTH_BIT;
		depth_range.baseArrayLayer        = layer;
		depth_range.layerCount            = 1;
		view_create_info.subresourceRange = native_range;
		if (depth_range.aspectMask != 0)
		{
			view_create_info.subresourceRange = depth_range;
			vkCreateImageView(GraphicsContext::instance()->getLogicalDevice(), &view_create_info, nullptr, &views.depth);
		}

		// Stencil
		auto stencil_range = getSubresourceRange();
		stencil_range.aspectMask &= VK_IMAGE_ASPECT_STENCIL_BIT;
		stencil_range.baseArrayLayer      = layer;
		stencil_range.layerCount          = 1;
		view_create_info.subresourceRange = native_range;
		if (stencil_range.aspectMask != 0)
		{
			view_create_info.subresourceRange = stencil_range;
			vkCreateImageView(GraphicsContext::instance()->getLogicalDevice(), &view_create_info, nullptr, &views.stencil);
		}

		layer++;
	}
}

void Image::create()
{
}

void Image::destroy()
{
	if (m_handle)
	{
		if (m_allocation)
		{
			vmaDestroyImage(GraphicsContext::instance()->getLogicalDevice().getAllocator(), m_handle, m_allocation);
		}

		vkDestroyImageView(GraphicsContext::instance()->getLogicalDevice(), m_views.native, nullptr);
		if (m_views.depth)
		{
			vkDestroyImageView(GraphicsContext::instance()->getLogicalDevice(), m_views.depth, nullptr);
		}
		if (m_views.stencil)
		{
			vkDestroyImageView(GraphicsContext::instance()->getLogicalDevice(), m_views.stencil, nullptr);
		}

		for (auto &views : m_layer_views)
		{
			vkDestroyImageView(GraphicsContext::instance()->getLogicalDevice(), views.native, nullptr);
			if (views.depth)
			{
				vkDestroyImageView(GraphicsContext::instance()->getLogicalDevice(), views.depth, nullptr);
			}
			if (views.stencil)
			{
				vkDestroyImageView(GraphicsContext::instance()->getLogicalDevice(), views.stencil, nullptr);
			}
		}
	}

	m_handle        = VK_NULL_HANDLE;
	m_allocation    = VK_NULL_HANDLE;
	m_views.native  = VK_NULL_HANDLE;
	m_views.depth   = VK_NULL_HANDLE;
	m_views.stencil = VK_NULL_HANDLE;
	m_layer_views.clear();
}
}        // namespace Ilum