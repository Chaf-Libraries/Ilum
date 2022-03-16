#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class LogicalDevice;
class CommandPool;
class CommandBuffer;

enum class ImageViewType
{
	Native = 0,
	Cube,
	ArrayCube,
	Depth_Only,
	Stencil_Only
};

class Image
{
  public:
	Image() = default;

	Image(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VmaMemoryUsage memory_usage, bool mip_level = false, uint32_t layer_count = 1);

	Image(VkImage image, uint32_t width, uint32_t height, VkFormat format);

	Image(const Image &) = delete;

	Image &operator=(const Image &other) = delete;

	Image(Image &&other) noexcept;

	Image &operator=(Image &&other) noexcept;

	~Image();

	const VkImageView &getView(ImageViewType type = ImageViewType::Native) const;

	const VkImageView &getView(uint32_t layer, ImageViewType type = ImageViewType::Native) const;

	uint32_t getWidth() const;

	uint32_t getHeight() const;

	uint32_t getMipWidth(uint32_t mip_level) const;

	uint32_t getMipHeight(uint32_t mip_level) const;

	uint32_t getMipLevelCount() const;

	uint32_t getLayerCount() const;

	VkFormat getFormat() const;

	const VkImage &getImage() const;

	operator const VkImage &() const;

	VkImageSubresourceLayers getSubresourceLayers(uint32_t mip_level = 0, uint32_t layer = 0) const;

	VkImageSubresourceRange getSubresourceRange() const;

	bool isDepth() const;

	bool isStencil() const;

  public:
	static VkImageAspectFlags format_to_aspect(VkFormat format);

	static VkImageLayout usage_to_layout(VkImageUsageFlagBits usage);

	static VkAccessFlags usage_to_access(VkImageUsageFlagBits usage);

	static VkPipelineStageFlags usage_to_stage(VkImageUsageFlagBits usage);

  private:
	void createImageViews();

	void create();

	void destroy();

  private:
	struct ImageViews
	{
		VkImageView native     = VK_NULL_HANDLE;
		VkImageView cube     = VK_NULL_HANDLE;
		VkImageView array_cube  = VK_NULL_HANDLE;
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
}        // namespace Ilum