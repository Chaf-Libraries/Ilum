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
	vmaCreateImage(p_device->GetAllocator(), &image_create_info, &allocation_create_info, &m_handle, &m_allocation, nullptr);
}

Texture::Texture(RHIDevice *device, const TextureDesc &desc, VkImage handle) :
    p_device(device), m_handle(handle), m_desc(desc)
{
}

Texture::~Texture()
{
	vkDeviceWaitIdle(p_device->GetDevice());
	if (m_handle && m_allocation)
	{
		vmaDestroyImage(p_device->GetAllocator(), m_handle, m_allocation);
	}

	for (auto &[hash, view] : m_views)
	{
		vkDestroyImageView(p_device->GetDevice(), view, nullptr);
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
		vkSetDebugUtilsObjectNameEXT(p_device->GetDevice(), &name_info);
	}
}

VkImageView Texture::GetView(const TextureViewDesc &desc)
{
	size_t hash = desc.Hash();
	if (m_views.find(hash) != m_views.end())
	{
		return m_views.at(hash);
	}

	VkImageViewCreateInfo view_create_info           = {};
	view_create_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	view_create_info.format                          = m_desc.format;
	view_create_info.image                           = m_handle;
	view_create_info.subresourceRange.aspectMask     = desc.aspect;
	view_create_info.subresourceRange.baseArrayLayer = desc.base_array_layer;
	view_create_info.subresourceRange.baseMipLevel   = desc.base_mip_level;
	view_create_info.subresourceRange.layerCount     = desc.layer_count;
	view_create_info.subresourceRange.levelCount     = desc.layer_count;
	view_create_info.viewType                        = desc.view_type;

	m_views[hash] = VK_NULL_HANDLE;
	vkCreateImageView(p_device->GetDevice(), &view_create_info, nullptr, &m_views[hash]);

	return m_views.at(hash);
}

const TextureDesc &Texture::GetDesc() const
{
	return m_desc;
}

TextureState::TextureState(VkImageUsageFlagBits usage)
{
	switch (usage)
	{
		case VK_IMAGE_USAGE_TRANSFER_SRC_BIT:
			layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			break;
		case VK_IMAGE_USAGE_TRANSFER_DST_BIT:
			layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			break;
		case VK_IMAGE_USAGE_SAMPLED_BIT:
			layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			break;
		case VK_IMAGE_USAGE_STORAGE_BIT:
			layout = VK_IMAGE_LAYOUT_GENERAL;
			break;
		case VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT:
			layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			break;
		case VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT:
			layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			break;
		case VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT:
			layout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
			break;
		case VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR:
			layout = VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;
			break;
		default:
			layout = VK_IMAGE_LAYOUT_UNDEFINED;
			break;
	}

	switch (usage)
	{
		case VK_IMAGE_USAGE_TRANSFER_SRC_BIT:
			access_mask = VK_ACCESS_TRANSFER_READ_BIT;
			break;
		case VK_IMAGE_USAGE_TRANSFER_DST_BIT:
			access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
			break;
		case VK_IMAGE_USAGE_SAMPLED_BIT:
			access_mask = VK_ACCESS_SHADER_READ_BIT;
			break;
		case VK_IMAGE_USAGE_STORAGE_BIT:
			access_mask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			break;
		case VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT:
			access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			break;
		case VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT:
			access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			break;
		case VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT:
			access_mask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
			break;
		case VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR:
			access_mask = VK_ACCESS_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR;
			break;
		default:
			access_mask = VK_ACCESS_NONE_KHR;
			break;
	}

	switch (usage)
	{
		case VK_IMAGE_USAGE_TRANSFER_SRC_BIT:
			stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			break;
		case VK_IMAGE_USAGE_TRANSFER_DST_BIT:
			stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			break;
		case VK_IMAGE_USAGE_SAMPLED_BIT:
			stage = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
			break;
		case VK_IMAGE_USAGE_STORAGE_BIT:
			stage = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
			break;
		case VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT:
			stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			break;
		case VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT:
			stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
			break;
		case VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT:
			stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			break;
		case VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR:
			stage = VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
			break;
		default:
			stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			break;
	}
}

}        // namespace Ilum