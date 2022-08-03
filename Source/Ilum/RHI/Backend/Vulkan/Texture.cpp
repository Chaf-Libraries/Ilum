#include "Texture.hpp"
#include "Definitions.hpp"

#include "Device.hpp"

namespace Ilum::Vulkan
{
TextureState TextureState::Create(RHITextureState state)
{
	TextureState vk_state = {};

	switch (state)
	{
		case RHITextureState::TransferSource:
			vk_state.layout      = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			vk_state.access_mask = VK_ACCESS_TRANSFER_READ_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_TRANSFER_BIT;
			break;
		case RHITextureState::TransferDest:
			vk_state.layout      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			vk_state.access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_TRANSFER_BIT;
			break;
		case RHITextureState::ShaderResource:
			vk_state.layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			vk_state.access_mask = VK_ACCESS_SHADER_READ_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
			break;
		case RHITextureState::UnorderAccess:
			vk_state.layout      = VK_IMAGE_LAYOUT_GENERAL;
			vk_state.access_mask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
			break;
		case RHITextureState::RenderTarget:
			vk_state.layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			vk_state.access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			break;
		case RHITextureState::DepthWrite:
			vk_state.layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			vk_state.access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
			break;
		case RHITextureState::DepthRead:
			vk_state.layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			vk_state.access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			break;
		case RHITextureState::Present:
			vk_state.layout      = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
			vk_state.access_mask = VK_ACCESS_MEMORY_READ_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
			break;
		default:
			vk_state.layout      = VK_IMAGE_LAYOUT_UNDEFINED;
			vk_state.access_mask = VK_ACCESS_NONE_KHR;
			vk_state.stage       = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			break;
	}

	return vk_state;
}

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

Texture::Texture(RHIDevice *device, const TextureDesc &desc, VkImage image) :
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