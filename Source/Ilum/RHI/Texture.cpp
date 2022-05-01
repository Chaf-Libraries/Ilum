#include "Texture.hpp"
#include "Device.hpp"

namespace Ilum
{
Texture::Texture(RHIDevice *device, const TextureDesc &desc) :
    p_device(device), m_desc(desc)
{
	// Create VkImage
	VkImageType image_type = VK_IMAGE_TYPE_1D;
	if (desc.depth > 1)
	{
		image_type = VK_IMAGE_TYPE_3D;
	}
	if (desc.height > 1)
	{
		image_type = VK_IMAGE_TYPE_2D;
	}

	VkImageCreateInfo image_create_info = {};
	image_create_info.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_create_info.imageType         = VK_IMAGE_TYPE_2D;
	image_create_info.format            = desc.format;
	image_create_info.extent            = VkExtent3D{desc.width, desc.height, desc.depth};
	image_create_info.samples           = desc.sample_count;
	image_create_info.mipLevels         = desc.mips;
	image_create_info.arrayLayers       = desc.layers;
	image_create_info.tiling            = VK_IMAGE_TILING_OPTIMAL;
	image_create_info.usage             = desc.usage;
	image_create_info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
	image_create_info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;

	if (desc.layers % 6 == 0 && desc.width == desc.height && desc.depth == 1)
	{
		image_create_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	}

	VmaAllocationCreateInfo allocation_create_info = {};
	allocation_create_info.usage                   = VMA_MEMORY_USAGE_GPU_ONLY;
	vmaCreateImage(p_device->m_allocator, &image_create_info, &allocation_create_info, &m_handle, &m_allocation, nullptr);
}

Texture::Texture(RHIDevice *device, const TextureDesc &desc, VkImage handle) :
    p_device(device), m_handle(handle), m_desc(desc)
{
}

Texture::~Texture()
{
	if (m_handle && m_allocation)
	{
		vmaDestroyImage(p_device->m_allocator, m_handle, m_allocation);
	}
}

uint32_t Texture::GetWidth() const
{
	return m_desc.width;
}

uint32_t Texture::GetHeight() const
{
	return m_desc.height;
}

uint32_t Texture::GetDepth() const
{
	return m_desc.depth;
}

uint32_t Texture::GetMipLevels() const
{
	return m_desc.mips;
}

uint32_t Texture::GetLayerCount() const
{
	return m_desc.layers;
}

VkFormat Texture::GetFormat() const
{
	return m_desc.format;
}

VkImageUsageFlags Texture::GetUsage() const
{
	return m_desc.usage;
}

Texture::operator VkImage() const
{
	return m_handle;
}

void Texture::SetName(const std::string &name)
{
	if (name.empty())
	{
		VkDebugUtilsObjectNameInfoEXT name_info = {};
		name_info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
		name_info.pNext                         = nullptr;
		name_info.objectType                    = VK_OBJECT_TYPE_IMAGE;
		name_info.objectHandle                  = (uint64_t) m_handle;
		name_info.pObjectName                   = name.c_str();
		vkSetDebugUtilsObjectNameEXT(p_device->m_device, &name_info);
	}
}

TextureView::TextureView(RHIDevice *device, Texture *texture, const TextureViewDesc &desc) :
    p_device(device), m_desc(desc)
{
	VkImageViewCreateInfo view_create_info           = {};
	view_create_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	view_create_info.format                          = texture->GetFormat();
	view_create_info.image                           = *texture;
	view_create_info.subresourceRange.aspectMask     = desc.aspect;
	view_create_info.subresourceRange.baseArrayLayer = desc.base_array_layer;
	view_create_info.subresourceRange.baseMipLevel   = desc.base_mip_level;
	view_create_info.subresourceRange.layerCount     = desc.layer_count;
	view_create_info.subresourceRange.levelCount     = desc.layer_count;
	view_create_info.viewType                        = desc.view_type;

	vkCreateImageView(p_device->m_device, &view_create_info, nullptr, &m_handle);
}

TextureView::~TextureView()
{
	if (m_handle)
	{
		vkDeviceWaitIdle(p_device->m_device);
		vkDestroyImageView(p_device->m_device, m_handle, nullptr);
	}
}

TextureView::operator VkImageView() const
{
	return m_handle;
}

void TextureView::SetName(const std::string &name)
{
	VkDebugUtilsObjectNameInfoEXT name_info = {};
	name_info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
	name_info.pNext                         = nullptr;
	name_info.objectType                    = VK_OBJECT_TYPE_IMAGE_VIEW;
	name_info.objectHandle                  = (uint64_t) m_handle;
	name_info.pObjectName                   = name.c_str();
	vkSetDebugUtilsObjectNameEXT(p_device->m_device, &name_info);
}
}        // namespace Ilum