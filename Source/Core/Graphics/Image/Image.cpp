#include "Image.hpp"

#include "Core/Device/LogicalDevice.hpp"
#include "Core/Device/PhysicalDevice.hpp"

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
}        // namespace Ilum