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
	Depth_Only,
	Stencil_Only
};

//class Image
//{
//  public:
//	Image(
//	    const VkExtent3D &    extent,
//	    VkFormat              format,
//	    VkImageUsageFlags     usage,
//	    uint32_t              mip_levels,
//	    uint32_t              array_layers,
//	    VkImageLayout         layout,
//	    VkFilter              filter,
//	    VkSamplerAddressMode  address_mode,
//	    VkSampleCountFlagBits samples);
//
//	~Image();
//
//	const VkExtent3D &getExtent() const;
//
//	uint32_t getWidth() const;
//
//	uint32_t getHeight() const;
//
//	VkSampleCountFlagBits getSamples() const;
//
//	VkImageUsageFlags getUsage() const;
//
//	VkFormat getFormat() const;
//
//	uint32_t getMipLevels() const;
//
//	uint32_t getArrayLevels() const;
//
//	VkFilter getFilter() const;
//
//	VkSamplerAddressMode getAddressMode() const;
//
//	VkImageLayout getImageLayout() const;
//
//	const VkImage &getImage() const;
//
//	const VkImageView &getView() const;
//
//	const VkSampler &getSampler() const;
//
//	const VkDescriptorImageInfo &getDescriptor() const;
//
//  public:
//	static uint32_t getMipLevels(const VkExtent3D &extent);
//
//	static bool hasDepth(VkFormat format);
//
//	static bool hasStencil(VkFormat format);
//
//	static VkFormat findSupportedFormat(const std::vector<VkFormat> &formats, VkImageTiling tiling, VkFormatFeatureFlags features);
//
//  public:
//	static bool createImage(
//	    VkImage &             image,
//	    VmaAllocation &       allocation,
//	    const VkExtent3D &    extent,
//	    VkImageType           type,
//	    VkFormat              format,
//	    uint32_t              mip_levels,
//	    uint32_t              array_layers,
//	    VkSampleCountFlagBits samples,
//	    VkImageTiling         tiling,
//	    VkImageUsageFlags     usage,
//	    VmaMemoryUsage        memory_usage);
//
//	static bool createImageView(
//	    const VkImage &    image,
//	    VkImageView &      image_view,
//	    VkImageViewType    type,
//	    VkFormat           format,
//	    uint32_t           mip_levels,
//	    uint32_t           base_mip_level,
//	    uint32_t           layer_count,
//	    uint32_t           base_array_layer,
//	    VkImageAspectFlags image_aspect);
//
//	static bool createImageSampler(
//	    VkSampler &          sampler,
//	    VkFilter             filter,
//	    VkSamplerAddressMode address_mode,
//	    bool                 anisotropic,
//	    uint32_t             mip_levels);
//
//	static void createMipmaps(
//	    const VkImage &   image,
//	    const VkExtent3D &extent,
//	    VkFormat          format,
//	    VkImageLayout     dst_image_layout,
//	    uint32_t          mip_levels,
//	    uint32_t          base_array_layer,
//	    uint32_t          layer_count);
//
//	static void transitionImageLayout(
//	    const VkImage &    image,
//	    VkFormat           format,
//	    VkImageLayout      src_image_layout,
//	    VkImageLayout      dst_image_layout,
//	    VkImageAspectFlags image_aspect,
//	    uint32_t           mip_levels,
//	    uint32_t           base_mip_level,
//	    uint32_t           layer_count,
//	    uint32_t           base_array_layer);
//
//	static void insertImageMemoryBarrier(
//	    const CommandBuffer &command_buffer,
//	    const VkImage &      image,
//	    VkAccessFlags        src_access_mask,
//	    VkAccessFlags        dst_access_mask,
//	    VkImageLayout        old_image_layout,
//	    VkImageLayout        new_image_layout,
//	    VkPipelineStageFlags src_stage_mask,
//	    VkPipelineStageFlags dst_stage_mask,
//	    VkImageAspectFlags   image_aspect,
//	    uint32_t             mip_levels,
//	    uint32_t             base_mip_level,
//	    uint32_t             layer_count,
//	    uint32_t             base_array_layer);
//
//	static void copyBufferToImage(
//	    const VkBuffer &  buffer,
//	    const VkImage &   image,
//	    const VkExtent3D &extent,
//	    uint32_t          layer_count,
//	    uint32_t          base_array_layer);
//
//	static void copyImage(
//	    const VkImage &   src_image,
//	    VkImage &         dst_image,
//	    VmaAllocation &   dst_image_allocation,
//	    VkFormat          src_format,
//	    const VkExtent3D &extent,
//	    VkImageLayout     src_image_layout,
//	    uint32_t          mip_level,
//	    uint32_t          array_layer);
//
//  protected:
//	void updateDescriptor();
//
//  protected:
//	VkExtent3D            m_extent;
//	VkSampleCountFlagBits m_samples;
//	VkImageUsageFlags     m_usage;
//	VkFormat              m_format       = VK_FORMAT_UNDEFINED;
//	uint32_t              m_mip_levels   = 0;
//	uint32_t              m_array_layers = 0;
//
//	VkFilter             m_filter       = {};
//	VkSamplerAddressMode m_address_mode = {};
//
//	VkImageLayout m_layout = VK_IMAGE_LAYOUT_UNDEFINED;
//
//	VkImage     m_image   = VK_NULL_HANDLE;
//	VkImageView m_view    = VK_NULL_HANDLE;
//	VkSampler   m_sampler = VK_NULL_HANDLE;
//
//	VmaAllocation m_allocation = VK_NULL_HANDLE;
//
//	VkDescriptorImageInfo m_descriptor = {};
//};

class Image
{
  public:
	Image(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VmaMemoryUsage memory_usage, bool mip_level = false, uint32_t layer_count = 1);

	Image(VkImage image, uint32_t width, uint32_t height, VkFormat format);

	Image(Image &&other);

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

  public:
	static VkImageAspectFlags format_to_aspect(VkFormat format);

	static VkImageLayout usage_to_layout(VkImageUsageFlagBits usage);

	static VkAccessFlags usage_to_access(VkImageUsageFlagBits usage);

	static VkPipelineStageFlags usage_to_stage(VkImageUsageFlagBits usage);

  private:
	void createImageViews();

  private:
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
}        // namespace Ilum