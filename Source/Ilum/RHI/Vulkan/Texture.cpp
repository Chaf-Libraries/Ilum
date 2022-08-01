#include "Texture.hpp"
#include "Definitions.hpp"

#include "Device.hpp"

namespace Ilum::Vulkan
{
Texture::Texture(RHIDevice *device, const TextureDesc &desc) :
    RHITexture(device, desc)
{
	VkImageCreateInfo create_info = {};
	create_info.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	create_info.imageType         = VK_IMAGE_TYPE_2D;
	create_info.format            = ToVulkanFormat[desc.format];
	create_info.extent            = VkExtent3D{desc.width, desc.height, desc.depth};
	create_info.samples           = ToVulkanSampleCountFlag[desc.samples];
	create_info.mipLevels         = desc.mips;
	create_info.arrayLayers       = desc.layers;
	create_info.tiling            = VK_IMAGE_TILING_OPTIMAL;
	create_info.usage             = ToVulkanImageUsage(desc.usage);
	create_info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
	create_info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;

	// Render Target Setting
	if (desc.usage & RHITextureUsage::RenderTarget)
	{
		create_info.usage |= IsDepthFormat(desc.format) ?
		                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT :
                                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	}

	// Cubemap Setting
	if (desc.layers % 6 == 0 && desc.width == desc.height && desc.depth == 1)
	{
		create_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	}

	VmaAllocationCreateInfo allocation_create_info = {};
	allocation_create_info.usage                   = VMA_MEMORY_USAGE_GPU_ONLY;
	vmaCreateImage(static_cast<Device *>(p_device)->GetAllocator(), &create_info, &allocation_create_info, &m_handle, &m_allocation, nullptr);
}

Texture::Texture(RHIDevice *device, const TextureDesc &desc, VkImage image):
    RHITexture(device, desc), m_handle(image)
{
}

Texture::~Texture()
{
	if (m_handle && m_allocation)
	{
		vmaDestroyImage(static_cast<Device *>(p_device)->GetAllocator(), m_handle, m_allocation);
	}

	for (auto &[hash, view] : m_view_cache)
	{
		vkDestroyImageView(static_cast<Device *>(p_device)->GetDevice(), view, nullptr);
	}
}

VkImage Texture::GetHandle() const
{
	return m_handle;
}

VkImageView Texture::GetView(const TextureRange &range) const
{
	size_t hash = range.Hash();
	if (m_view_cache.find(hash) != m_view_cache.end())
	{
		return m_view_cache.at(hash);
	}

	VkImageViewCreateInfo view_create_info           = {};
	view_create_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	view_create_info.format                          = ToVulkanFormat[m_desc.format];
	view_create_info.image                           = m_handle;
	view_create_info.subresourceRange.aspectMask     = IsDepthFormat(m_desc.format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
	view_create_info.subresourceRange.baseArrayLayer = range.base_layer;
	view_create_info.subresourceRange.baseMipLevel   = range.base_mip;
	view_create_info.subresourceRange.layerCount     = range.layer_count;
	view_create_info.subresourceRange.levelCount     = range.mip_count;
	view_create_info.viewType                        = ToVulkanImageViewType[range.dimension];

	m_view_cache[hash] = VK_NULL_HANDLE;
	vkCreateImageView(static_cast<Device *>(p_device)->GetDevice(), &view_create_info, nullptr, &m_view_cache[hash]);

	return m_view_cache.at(hash);
}
}        // namespace Ilum::Vulkan