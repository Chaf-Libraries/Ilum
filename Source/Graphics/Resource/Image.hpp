#pragma once

#include "../Vulkan.hpp"

namespace Ilum::Graphics
{
class Device;

enum class ImageViewType
{
	Native = 0,
	Depth_Only,
	Stencil_Only
};

class Image
{
  public:
	Image(const Device &device);
	Image(const Device &device, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VmaMemoryUsage memory_usage, bool mip_level = false, uint32_t layer_count = 1);
	Image(const Device &device, VkImage image, uint32_t width, uint32_t height, VkFormat format);

	Image(const Image &) = delete;
	Image &operator=(const Image &other) = delete;
	Image(Image &&other) noexcept;
	Image &operator=(Image &&other) noexcept;

	~Image();

	operator const VkImage &() const;

	const VkImageView &GetView(ImageViewType type = ImageViewType::Native) const;
	const VkImageView &GetView(uint32_t layer, ImageViewType type = ImageViewType::Native) const;
	uint32_t GetWidth() const;
	uint32_t GetHeight() const;
	uint32_t GetMipWidth(uint32_t mip_level) const;
	uint32_t GetMipHeight(uint32_t mip_level) const;
	uint32_t GetMipLevelCount() const;
	uint32_t GetLayerCount() const;
	VkFormat GetFormat() const;
	const VkImage &GetHandle() const;
	VkImageSubresourceLayers GetSubresourceLayers(uint32_t mip_level = 0, uint32_t layer = 0) const;
	VkImageSubresourceRange GetSubresourceRange() const;

	bool IsDepth() const;

	bool IsStencil() const;

  public:
	static VkImageAspectFlags FormatToAspect(VkFormat format);

	static VkImageLayout UsageToLayout(VkImageUsageFlagBits usage);

	static VkAccessFlags UsageToAccess(VkImageUsageFlagBits usage);

	static VkPipelineStageFlags UsageToStage(VkImageUsageFlagBits usage);

  private:
	void CreateImageViews();

	void Destroy();

  private:
	const Device &m_device;

	struct ImageViews
	{
		VkImageView native  = VK_NULL_HANDLE;
		VkImageView depth   = VK_NULL_HANDLE;
		VkImageView stencil = VK_NULL_HANDLE;
	};

	VkImage                 m_handle = VK_NULL_HANDLE;
	ImageViews              m_views;
	std::vector<ImageViews> m_layer_views;
	VkExtent2D              m_extent          = {0u, 0u};
	uint32_t                m_mip_level_count = 1;
	uint32_t                m_layer_count     = 1;
	VkFormat                m_format          = VK_FORMAT_UNDEFINED;
	VmaAllocation           m_allocation      = VK_NULL_HANDLE;
};

using ImageReference = std::reference_wrapper<const Image>;

struct ImageInfo
{
	ImageReference       handle;
	VkImageUsageFlagBits usage           = VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM;
	uint32_t             base_mip_level  = 0;
	uint32_t             base_layer      = 0;
	uint32_t             mip_level_count = 0;
	uint32_t             layer_count     = 0;
};
}        // namespace Ilum::Graphics