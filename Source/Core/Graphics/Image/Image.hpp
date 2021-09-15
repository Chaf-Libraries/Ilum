#pragma once

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
class LogicalDevice;

class Image
{
  public:
	Image(
	    const LogicalDevice & logical_device,
	    const VkExtent3D &    extent,
	    VkFormat              format,
	    VkImageUsageFlags     usage,
	    uint32_t              mip_levels,
	    uint32_t              array_layers,
	    VkImageLayout         layout,
	    VkFilter              filter,
	    VkSamplerAddressMode  address_mode,
	    VkSampleCountFlagBits samples);

	~Image();

	const VkExtent3D &getExtent() const;

	uint32_t getWidth() const;

	uint32_t getHeight() const;

	VkSampleCountFlagBits getSamples() const;

	VkImageUsageFlags getUsage() const;

	VkFormat getFormat() const;

	uint32_t getMipLevels() const;

	uint32_t getArrayLevels() const;

	VkFilter getFilter() const;

	VkSamplerAddressMode getAddressMode() const;

	VkImageLayout getImageLayout() const;

	const VkImage &getImage() const;

	const VkImageView &getView() const;

	const VkSampler &getSampler() const;

  public:
	static bool createImage(
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
	    VmaMemoryUsage        memory_usage);

	static bool createImageView(
	    const LogicalDevice &logical_device,
	    const VkImage &      image,
	    VkImageView &        image_view,
	    VkImageViewType      type,
	    VkFormat             format,
	    uint32_t             mip_levels,
	    uint32_t             base_mip_level,
	    uint32_t             layer_count,
	    uint32_t             base_array_layer,
	    VkImageAspectFlags   image_aspect);

	static bool createImageSampler(
	    const LogicalDevice &logical_device,
	    VkSampler &          sampler,
	    VkFilter             filter,
	    VkSamplerAddressMode address_mode,
	    bool                 anisotropic,
	    uint32_t             mip_levels);

  protected:
	const LogicalDevice &m_logical_device;

	VkExtent3D            m_extent;
	VkSampleCountFlagBits m_samples;
	VkImageUsageFlags     m_usage;
	VkFormat              m_format       = VK_FORMAT_UNDEFINED;
	uint32_t              m_mip_levels   = 0;
	uint32_t              m_array_layers = 0;

	VkFilter             m_filter       = {};
	VkSamplerAddressMode m_address_mode = {};

	VkImageLayout m_layout = VK_IMAGE_LAYOUT_UNDEFINED;

	VkImage     m_image   = VK_NULL_HANDLE;
	VkImageView m_view    = VK_NULL_HANDLE;
	VkSampler   m_sampler = VK_NULL_HANDLE;

	VmaAllocation m_allocation = VK_NULL_HANDLE;
};
}        // namespace Ilum