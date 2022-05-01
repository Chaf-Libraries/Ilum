#pragma once

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
};

struct TextureViewDesc
{
	VkImageViewType    view_type = VK_IMAGE_VIEW_TYPE_MAX_ENUM;
	VkImageAspectFlags aspect;
	uint32_t           base_mip_level   = 0;
	uint32_t           level_count      = 0;
	uint32_t           base_array_layer = 0;
	uint32_t           layer_count      = 0;
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

  private:
	RHIDevice    *p_device = nullptr;
	TextureDesc   m_desc;
	VkImage       m_handle = VK_NULL_HANDLE;
	VmaAllocation m_allocation = VK_NULL_HANDLE;
};

class TextureView
{
  public:
	TextureView(RHIDevice *device, Texture *texture, const TextureViewDesc &desc);
	~TextureView();

	operator VkImageView() const;

	void SetName(const std::string &name);

  private:
	RHIDevice      *p_device = nullptr;
	TextureViewDesc m_desc;
	VkImageView     m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum