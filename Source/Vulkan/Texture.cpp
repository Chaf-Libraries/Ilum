#include "Texture.hpp"
#include "Device.hpp"
#include "RenderContext.hpp"

#include <Core/Hash.hpp>

namespace Ilum::Vulkan
{
inline VkImageAspectFlags FormatToAspect(VkFormat format)
{
	switch (format)
	{
		case VK_FORMAT_D16_UNORM:
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

inline VkImageType ExtentToType(VkExtent3D extent)
{
	uint32_t dim_num = 0;

	if (extent.width >= 1)
	{
		dim_num++;
	}
	if (extent.height >= 1)
	{
		dim_num++;
	}
	if (extent.depth >= 1)
	{
		dim_num++;
	}

	switch (dim_num)
	{
		case 1:
			return VK_IMAGE_TYPE_1D;
		case 2:
			return VK_IMAGE_TYPE_2D;
		case 3:
			return VK_IMAGE_TYPE_3D;
		default:
			throw std::runtime_error("No image type found.");
	}

	return VK_IMAGE_TYPE_MAX_ENUM;
}

Image::Image(const VkExtent3D &       extent,
             VkFormat                 format,
             VkImageUsageFlagBits     image_usage,
             VmaMemoryUsage           memory_usage,
             uint32_t                 mip_levels,
             uint32_t                 array_layers,
             VkSampleCountFlagBits    sample_count,
             VkImageTiling            tiling,
             VkImageCreateFlags       flags,
             std::vector<QueueFamily> queue_families) :
    m_extent(extent),
    m_format(format),
    m_usage(image_usage),
    m_mip_levels(mip_levels),
    m_array_layers(array_layers),
    m_sample_count(sample_count),
    m_tiling(tiling)
{
	VkImageCreateInfo image_info = {};
	image_info.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_info.flags             = flags;
	image_info.imageType         = ExtentToType(extent);
	image_info.format            = format;
	image_info.extent            = extent;
	image_info.mipLevels         = mip_levels;
	image_info.arrayLayers       = array_layers;
	image_info.samples           = sample_count;
	image_info.tiling            = tiling;
	image_info.usage             = image_usage;

	if (queue_families.size() > 0)
	{
		std::vector<uint32_t> queue_family_indices;
		for (auto &queue_family : queue_families)
		{
			queue_family_indices.push_back(RenderContext::GetDevice().GetQueueFamily(queue_family));
		}

		image_info.sharingMode           = VK_SHARING_MODE_CONCURRENT;
		image_info.queueFamilyIndexCount = static_cast<uint32_t>(queue_family_indices.size());
		image_info.pQueueFamilyIndices   = queue_family_indices.data();
	}

	VmaAllocationCreateInfo memory_info = {};
	memory_info.usage                   = memory_usage;

	if (image_usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)
	{
		memory_info.preferredFlags = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
	}

	vmaCreateImage(RenderContext::GetDevice().GetAllocator(),
	               &image_info, &memory_info,
	               &m_handle, &m_allocation,
	               nullptr);
}

Image::~Image()
{
	// Destroy all view
	for (auto &[hash, view] : m_views)
	{
		vkDestroyImageView(RenderContext::GetDevice(), view, nullptr);
	}

	// Destroy handle
	if (m_handle && m_allocation)
	{
		Unmap();
		vmaDestroyImage(RenderContext::GetDevice().GetAllocator(), m_handle, m_allocation);
	}
}

uint8_t *Image::Map()
{
	if (!m_mapped_data)
	{
		if (m_tiling != VK_IMAGE_TILING_LINEAR)
		{
			LOG_WARN("Mapping image memory that is not linear");
		}
		vmaMapMemory(RenderContext::GetDevice().GetAllocator(), m_allocation, reinterpret_cast<void **>(&m_mapped_data));
		m_mapped = true;
	}
	return m_mapped_data;
}

void Image::Unmap()
{
	if (m_mapped)
	{
		vmaUnmapMemory(RenderContext::GetDevice().GetAllocator(), m_allocation);
		m_mapped_data = nullptr;
		m_mapped      = false;
	}
}

Image::operator const VkImage &() const
{
	return m_handle;
}

const VkImage &Image::GetHandle() const
{
	return m_handle;
}

VkFormat Image::GetFormat() const
{
	return m_format;
}

const VkExtent3D &Image::GetExtent() const
{
	return m_extent;
}

VkImageUsageFlags Image::GetUsage() const
{
	return m_usage;
}

VkImageTiling Image::GetTiling() const
{
	return m_tiling;
}

VkSampleCountFlagBits Image::GetSampleCount() const
{
	return m_sample_count;
}

uint32_t Image::GetMipLevelCount() const
{
	return m_mip_levels;
}

uint32_t Image::GetArrayLayerCount() const
{
	return m_array_layers;
}

VkImageSubresource Image::GetSubresource() const
{
	VkImageSubresource subresource = {};
	subresource.aspectMask         = FormatToAspect(m_format);
	subresource.mipLevel           = m_mip_levels;
	subresource.arrayLayer         = m_array_layers;

	return subresource;
}

VkImageSubresourceLayers Image::GetSubresourceLayers(uint32_t mip_level, uint32_t base_layer, uint32_t layer_count) const
{
	assert(mip_level <= m_mip_levels &&
	       base_layer < m_array_layers &&
	       layer_count <= m_array_layers);

	VkImageSubresourceLayers subresource_layer = {};
	subresource_layer.aspectMask               = FormatToAspect(m_format);
	subresource_layer.mipLevel                 = mip_level;
	subresource_layer.baseArrayLayer           = base_layer;
	subresource_layer.layerCount               = layer_count;

	return subresource_layer;
}

VkImageSubresourceRange Image::GetSubresourceRange(uint32_t base_mip_level, uint32_t mip_level_count, uint32_t base_layer, uint32_t layer_count) const
{
	assert(base_mip_level < m_mip_levels &&
	       mip_level_count <= m_mip_levels &&
	       base_layer < m_array_layers &&
	       layer_count <= m_array_layers);

	VkImageSubresourceRange subresource_range = {};
	subresource_range.aspectMask              = FormatToAspect(m_format);
	subresource_range.baseMipLevel            = base_mip_level;
	subresource_range.levelCount              = mip_level_count;
	subresource_range.baseArrayLayer          = base_layer;
	subresource_range.layerCount              = layer_count;

	return subresource_range;
}

const VkImageView &Image::GetView(VkImageViewType view_type,
                                  uint32_t base_mip_level, uint32_t base_array_layer,
                                  uint32_t mip_level_count, uint32_t array_layer_count)
{
	size_t hash = 0;
	Core::HashCombine(hash, static_cast<size_t>(view_type));
	Core::HashCombine(hash, base_mip_level);
	Core::HashCombine(hash, base_array_layer);
	Core::HashCombine(hash, mip_level_count);
	Core::HashCombine(hash, array_layer_count);

	if (m_views.find(hash) != m_views.end())
	{
		return m_views[hash];
	}

	auto subresource_range = GetSubresourceRange(base_mip_level, mip_level_count, base_array_layer, array_layer_count);

	// We don't consider stencil view
	subresource_range.aspectMask = subresource_range.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT ?
                                       VK_IMAGE_ASPECT_DEPTH_BIT :
                                       subresource_range.aspectMask;

	VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
	view_info.image            = m_handle;
	view_info.viewType         = view_type;
	view_info.format           = m_format;
	view_info.subresourceRange = subresource_range;

	VkImageView view = VK_NULL_HANDLE;

	vkCreateImageView(RenderContext::GetDevice(), &view_info, nullptr, &view);

	m_views.emplace(hash, view);

	return view;
}
bool Image::IsDepth() const
{
	return FormatToAspect(m_format) & VK_IMAGE_ASPECT_DEPTH_BIT;
}

bool Image::IsStencil() const
{
	return FormatToAspect(m_format) & VK_IMAGE_ASPECT_STENCIL_BIT;
}

Sampler::Sampler(VkFilter min_filter, VkFilter mag_filter, VkSamplerAddressMode address_mode, VkFilter mip_filter)
{
	VkSamplerCreateInfo sampler_info = {};
	sampler_info.sType               = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler_info.minFilter           = min_filter;
	sampler_info.magFilter           = mag_filter;
	sampler_info.addressModeU        = address_mode;
	sampler_info.addressModeV        = address_mode;
	sampler_info.addressModeW        = address_mode;
	sampler_info.mipLodBias          = 0;
	sampler_info.minLod              = 0;
	sampler_info.maxLod              = 1000;

	vkCreateSampler(RenderContext::GetDevice(), &sampler_info, nullptr, &m_handle);
}

Sampler::~Sampler()
{
	if (m_handle)
	{
		vkDestroySampler(RenderContext::GetDevice(), m_handle, nullptr);
	}
}

Sampler::operator const VkSampler &() const
{
	return m_handle;
}

const VkSampler &Sampler::GetHandle() const
{
	return m_handle;
}
}        // namespace Ilum::Vulkan