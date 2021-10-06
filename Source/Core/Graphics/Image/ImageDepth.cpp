#include "ImageDepth.hpp"

namespace Ilum
{
inline static const std::vector<VkFormat> DEPTH_FORMATS = {
    VK_FORMAT_D32_SFLOAT_S8_UINT,
    VK_FORMAT_D32_SFLOAT,
    VK_FORMAT_D24_UNORM_S8_UINT,
    VK_FORMAT_D16_UNORM_S8_UINT,
    VK_FORMAT_D16_UNORM};

ImageDepth::ImageDepth(const uint32_t width, const uint32_t height, VkSampleCountFlagBits samples) :
    Image(
        {width, height, 1},
        findSupportedFormat(DEPTH_FORMATS, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT),
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        1,
        1,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        VK_FILTER_LINEAR,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        samples)
{
    if (m_format == VK_FORMAT_UNDEFINED)
    {
		throw std::runtime_error("No depth stencil format is supported!");
    }

    VkImageAspectFlags aspect_mask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (hasStencil(m_format))
    {
		aspect_mask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }

    createImage(m_image, m_allocation, m_extent, VK_IMAGE_TYPE_2D, m_format, 1, 1, samples, VK_IMAGE_TILING_OPTIMAL, m_usage, VMA_MEMORY_USAGE_GPU_ONLY);
	createImageSampler(m_sampler, m_filter, m_address_mode, false, 1);
	createImageView(m_image, m_view, VK_IMAGE_VIEW_TYPE_2D, m_format, 1, 0, 1, 0, VK_IMAGE_ASPECT_DEPTH_BIT);
	transitionImageLayout(m_image, m_format, VK_IMAGE_LAYOUT_UNDEFINED, m_layout, aspect_mask, 1, 0, 1, 0);
}

}        // namespace Ilum