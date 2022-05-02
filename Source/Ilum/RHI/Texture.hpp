#pragma once

#include <Core/Hash.hpp>

#include <volk.h>

#include <vk_mem_alloc.h>

#include <memory>
#include <string>

namespace Ilum
{
class RHIDevice;
class Texture;
class TextureView;

struct TextureDesc
{
	uint32_t              width        = 1;
	uint32_t              height       = 1;
	uint32_t              depth        = 1;
	uint32_t              mips         = 1;
	uint32_t              layers       = 1;
	VkSampleCountFlagBits sample_count = VK_SAMPLE_COUNT_1_BIT;
	VkFormat              format       = VK_FORMAT_UNDEFINED;
	VkImageUsageFlags     usage        = VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM;

	size_t Hash() const
	{
		size_t hash = 0;
		HashCombine(hash, width);
		HashCombine(hash, height);
		HashCombine(hash, depth);
		HashCombine(hash, mips);
		HashCombine(hash, layers);
		HashCombine(hash, sample_count);
		HashCombine(hash, format);
		HashCombine(hash, usage);
		return hash;
	}
};

struct TextureViewDesc
{
	VkImageViewType    view_type = VK_IMAGE_VIEW_TYPE_MAX_ENUM;
	VkImageAspectFlags aspect;
	uint32_t           base_mip_level   = 0;
	uint32_t           level_count      = 0;
	uint32_t           base_array_layer = 0;
	uint32_t           layer_count      = 0;

	size_t Hash() const
	{
		size_t hash = 0;
		HashCombine(hash, view_type);
		HashCombine(hash, aspect);
		HashCombine(hash, base_mip_level);
		HashCombine(hash, level_count);
		HashCombine(hash, base_array_layer);
		HashCombine(hash, layer_count);
		return hash;
	}
};

struct TextureState
{
	VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
	VkAccessFlags access_mask = VK_ACCESS_NONE;
	VkPipelineStageFlags stage       = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;

	TextureState(VkImageUsageFlagBits usage = VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM);
};

class Texture
{
	friend class RHIDevice;

  public:
	Texture(RHIDevice *device, const TextureDesc &desc);
	// Swaphcain Texture
	Texture(RHIDevice *device, const TextureDesc &desc, VkImage handle);
	~Texture();

	Texture(const Texture &) = delete;
	Texture &operator=(const Texture &) = delete;
	Texture(Texture &&other)            = delete;
	Texture &operator=(Texture &&other) = delete;

	uint32_t          GetWidth() const;
	uint32_t          GetHeight() const;
	uint32_t          GetDepth() const;
	uint32_t          GetMipLevels() const;
	uint32_t          GetLayerCount() const;
	VkFormat          GetFormat() const;
	VkImageUsageFlags GetUsage() const;

	operator VkImage() const;

	void SetName(const std::string &name);

	VkImageView GetView(const TextureViewDesc &desc);

  private:
	RHIDevice    *p_device = nullptr;
	TextureDesc   m_desc;
	VkImage       m_handle     = VK_NULL_HANDLE;
	VmaAllocation m_allocation = VK_NULL_HANDLE;

	std::unordered_map<size_t, VkImageView> m_views;
};

using TextureReference = std::reference_wrapper<Texture>;
}        // namespace Ilum