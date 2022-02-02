#pragma once

#include "Vulkan.hpp"

#include <map>

namespace Ilum::Vulkan
{
class Image
{
  public:
	Image(const VkExtent3D &       extent,
	      VkFormat                 format,
	      VkImageUsageFlagBits     image_usage,
	      VmaMemoryUsage           memory_usage,
	      uint32_t                 mip_levels     = 1,
	      uint32_t                 array_layers   = 1,
	      VkSampleCountFlagBits    sample_count   = VK_SAMPLE_COUNT_1_BIT,
	      VkImageTiling            tiling         = VK_IMAGE_TILING_OPTIMAL,
	      VkImageCreateFlags       flags          = 0,
	      std::vector<QueueFamily> queue_families = {});
	~Image();

	Image(const Image &) = delete;
	Image &operator=(const Image &) = delete;
	Image(Image &&)                 = delete;
	Image &operator=(Image &&) = delete;

	uint8_t *Map();
	void     Unmap();

	operator const VkImage &() const;

	const VkImage &          GetHandle() const;
	VkFormat                 GetFormat() const;
	const VkExtent3D &       GetExtent() const;
	VkImageUsageFlags        GetUsage() const;
	VkImageTiling            GetTiling() const;
	VkSampleCountFlagBits    GetSampleCount() const;
	uint32_t                 GetMipLevelCount() const;
	uint32_t                 GetArrayLayerCount() const;
	VkImageSubresource       GetSubresource() const;
	VkImageSubresourceLayers GetSubresourceLayers(uint32_t mip_level, uint32_t base_layer, uint32_t layer_count = 1) const;
	VkImageSubresourceRange  GetSubresourceRange(uint32_t base_mip_level, uint32_t mip_level_count, uint32_t base_layer, uint32_t layer_count) const;
	const VkImageView &      GetView(VkImageViewType view_type,
	                                 uint32_t base_mip_level, uint32_t base_array_layer,
	                                 uint32_t mip_level_count, uint32_t array_layer_count);

	bool IsDepth() const;
	bool IsStencil() const;

  private:
	VkImage               m_handle       = VK_NULL_HANDLE;
	VkFormat              m_format       = {};
	VmaAllocation         m_allocation   = VK_NULL_HANDLE;
	VkExtent3D            m_extent       = {};
	VkImageUsageFlags     m_usage        = {};
	VkImageTiling         m_tiling       = {};
	VkImageSubresource    m_subresource  = {};
	VkSampleCountFlagBits m_sample_count = {};
	uint32_t              m_mip_levels   = 0;
	uint32_t              m_array_layers = 0;

	uint8_t *m_mapped_data = nullptr;
	bool     m_mapped      = false;

	std::map<size_t, VkImageView> m_views;
};

class Sampler
{
  public:
	Sampler(VkFilter min_filter, VkFilter mag_filter, VkSamplerAddressMode address_mode, VkFilter mip_filter);
	~Sampler();

	Sampler(const Sampler &) = delete;
	Sampler &operator=(const Sampler &) = delete;
	Sampler(Sampler &&)                 = delete;
	Sampler &operator=(Sampler &&) = delete;

	operator const VkSampler &() const;

	const VkSampler &GetHandle() const;

  private:
	VkSampler m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum::Vulkan