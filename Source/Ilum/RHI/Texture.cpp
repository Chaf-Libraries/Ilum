#include "Texture.hpp"
#include "Buffer.hpp"
#include "Command.hpp"
#include "Device.hpp"

#include <Core/Path.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

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

Texture::Texture(RHIDevice *device, const std::string &filename) :
    p_device(device)
{
	// Load from disk
	int32_t       width = 0, height = 0, channel = 0;
	const int32_t req_channel = 4;

	void  *data = nullptr;
	size_t size = 0;

	if (stbi_is_hdr(filename.c_str()))
	{
		data          = stbi_loadf(filename.c_str(), &width, &height, &channel, req_channel);
		size          = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(float);
		m_desc.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	}
	else if (stbi_is_16_bit(filename.c_str()))
	{
		data          = stbi_load_16(filename.c_str(), &width, &height, &channel, req_channel);
		size          = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(uint16_t);
		m_desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
	}
	else
	{
		data          = stbi_load(filename.c_str(), &width, &height, &channel, req_channel);
		size          = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(uint8_t);
		m_desc.format = VK_FORMAT_R8G8B8A8_UNORM;
	}

	m_desc.width  = static_cast<uint32_t>(width);
	m_desc.height = static_cast<uint32_t>(height);
	m_desc.mips   = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height))) + 1);
	m_desc.usage  = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

	BufferDesc buffer_desc   = {};
	buffer_desc.size         = size;
	buffer_desc.buffer_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	buffer_desc.memory_usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	Buffer staging_buffer(p_device, buffer_desc);
	std::memcpy(staging_buffer.Map(), data, buffer_desc.size);
	staging_buffer.Flush(buffer_desc.size);
	staging_buffer.Unmap();

	// Create VkImage
	VkImageCreateInfo image_create_info = {};
	image_create_info.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_create_info.imageType         = VK_IMAGE_TYPE_2D;
	image_create_info.format            = m_desc.format;
	image_create_info.extent            = VkExtent3D{m_desc.width, m_desc.height, m_desc.depth};
	image_create_info.samples           = m_desc.sample_count;
	image_create_info.mipLevels         = m_desc.mips;
	image_create_info.arrayLayers       = m_desc.layers;
	image_create_info.tiling            = VK_IMAGE_TILING_OPTIMAL;
	image_create_info.usage             = m_desc.usage;
	image_create_info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
	image_create_info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;

	VmaAllocationCreateInfo allocation_create_info = {};
	allocation_create_info.usage                   = VMA_MEMORY_USAGE_GPU_ONLY;
	vmaCreateImage(p_device->GetAllocator(), &image_create_info, &allocation_create_info, &m_handle, &m_allocation, nullptr);

	// Transfer
	BufferCopyInfo buffer_info = {};
	buffer_info.buffer         = &staging_buffer;

	TextureCopyInfo tex_info            = {};
	tex_info.texture                    = this;
	tex_info.subresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	tex_info.subresource.baseArrayLayer = 0;
	tex_info.subresource.layerCount     = 1;
	tex_info.subresource.mipLevel       = 0;

	auto &cmd_buffer = p_device->RequestCommandBuffer();
	cmd_buffer.Begin();
	cmd_buffer.Transition(this, TextureState{}, TextureState{VK_IMAGE_USAGE_TRANSFER_DST_BIT}, VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, m_desc.mips, 0, m_desc.layers});
	cmd_buffer.CopyBufferToImage(buffer_info, tex_info);
	cmd_buffer.GenerateMipmap(this, TextureState{VK_IMAGE_USAGE_TRANSFER_DST_BIT}, VK_FILTER_LINEAR);
	cmd_buffer.Transition(this, TextureState{VK_IMAGE_USAGE_TRANSFER_DST_BIT}, TextureState{VK_IMAGE_USAGE_SAMPLED_BIT}, VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, m_desc.mips, 0, m_desc.layers});
	cmd_buffer.End();
	p_device->SubmitIdle(cmd_buffer);

	SetName(Path::GetInstance().GetFileName(filename, false));
}

Texture::Texture(RHIDevice *device, void *raw_data, int32_t raw_size) :
    p_device(device)
{
	// Load from buffer
	int32_t       width = 0, height = 0, channel = 0;
	const int32_t req_channel = 4;

	void  *data = nullptr;
	size_t size = 0;

	if (stbi_is_hdr_from_memory(static_cast<stbi_uc *>(raw_data), raw_size))
	{
		data          = stbi_loadf_from_memory(static_cast<stbi_uc *>(raw_data), raw_size, &width, &height, &channel, req_channel);
		size          = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(float);
		m_desc.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	}
	else if (stbi_is_16_bit_from_memory(static_cast<stbi_uc *>(raw_data), raw_size))
	{
		data          = stbi_load_16_from_memory(static_cast<stbi_uc *>(raw_data), raw_size, &width, &height, &channel, req_channel);
		size          = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(uint16_t);
		m_desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
	}
	else
	{
		data          = stbi_load_from_memory(static_cast<stbi_uc *>(raw_data), raw_size, &width, &height, &channel, req_channel);
		size          = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(uint8_t);
		m_desc.format = VK_FORMAT_R8G8B8A8_UNORM;
	}

	m_desc.width  = static_cast<uint32_t>(width);
	m_desc.height = static_cast<uint32_t>(height);
	m_desc.mips   = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height))) + 1);
	m_desc.usage  = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

	BufferDesc buffer_desc   = {};
	buffer_desc.size         = size;
	buffer_desc.buffer_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	buffer_desc.memory_usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	Buffer staging_buffer(p_device, buffer_desc);
	std::memcpy(staging_buffer.Map(), data, buffer_desc.size);
	staging_buffer.Flush(buffer_desc.size);
	staging_buffer.Unmap();

	// Create VkImage
	VkImageCreateInfo image_create_info = {};
	image_create_info.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_create_info.imageType         = VK_IMAGE_TYPE_2D;
	image_create_info.format            = m_desc.format;
	image_create_info.extent            = VkExtent3D{m_desc.width, m_desc.height, m_desc.depth};
	image_create_info.samples           = m_desc.sample_count;
	image_create_info.mipLevels         = m_desc.mips;
	image_create_info.arrayLayers       = m_desc.layers;
	image_create_info.tiling            = VK_IMAGE_TILING_OPTIMAL;
	image_create_info.usage             = m_desc.usage;
	image_create_info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
	image_create_info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;

	VmaAllocationCreateInfo allocation_create_info = {};
	allocation_create_info.usage                   = VMA_MEMORY_USAGE_GPU_ONLY;
	vmaCreateImage(p_device->GetAllocator(), &image_create_info, &allocation_create_info, &m_handle, &m_allocation, nullptr);

	// Transfer
	BufferCopyInfo buffer_info = {};
	buffer_info.buffer         = &staging_buffer;

	TextureCopyInfo tex_info            = {};
	tex_info.texture                    = this;
	tex_info.subresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	tex_info.subresource.baseArrayLayer = 0;
	tex_info.subresource.layerCount     = 1;
	tex_info.subresource.mipLevel       = 0;

	auto &cmd_buffer = p_device->RequestCommandBuffer();
	cmd_buffer.Begin();
	cmd_buffer.Transition(this, TextureState{}, TextureState{VK_IMAGE_USAGE_TRANSFER_DST_BIT}, VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, m_desc.mips, 0, m_desc.layers});
	cmd_buffer.CopyBufferToImage(buffer_info, tex_info);
	cmd_buffer.GenerateMipmap(this, TextureState{VK_IMAGE_USAGE_TRANSFER_DST_BIT}, VK_FILTER_LINEAR);
	cmd_buffer.Transition(this, TextureState{VK_IMAGE_USAGE_TRANSFER_DST_BIT}, TextureState{VK_IMAGE_USAGE_SAMPLED_BIT}, VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, m_desc.mips, 0, m_desc.layers});
	cmd_buffer.End();
	p_device->SubmitIdle(cmd_buffer);
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

uint32_t Texture::GetMipWidth(uint32_t level) const
{
	return std::max(m_desc.width, 1u << level) >> level;
}

uint32_t Texture::GetMipHeight(uint32_t level) const
{
	return std::max(m_desc.height, 1u << level) >> level;
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
	m_name = name;
}

const std::string &Texture::GetName() const
{
	return m_name;
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

bool Texture::IsDepth() const
{
	return m_desc.format == VK_FORMAT_D32_SFLOAT || m_desc.format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

bool Texture::IsStencil() const
{
	return m_desc.format == VK_FORMAT_D32_SFLOAT_S8_UINT;
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