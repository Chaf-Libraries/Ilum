#include "Image2D.hpp"

#include "Device/LogicalDevice.hpp"

#include "Graphics/Buffer/Buffer.h"

namespace Ilum
{
Image2D::Image2D(
    const uint32_t        width,
    const uint32_t        height,
    VkFormat              format,
    VkImageLayout         layout,
    VkImageUsageFlags     usage,
    VkFilter              filter,
    VkSamplerAddressMode  address_mode,
    VkSampleCountFlagBits samples,
    bool                  anisotropic,
    bool                  mipmap) :
    Image(
        VkExtent3D{width, height, 1u},
        format,
        usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        1,
        1,
        layout,
        filter,
        address_mode,
        samples),
    m_anisotropic(anisotropic),
    m_mipmap(mipmap),
    m_bytes_per_pixel(4)
{
	load();
}

Image2D::Image2D(
    scope<Bitmap> &&      bitmap,
    VkFormat              format,
    VkImageLayout         layout,
    VkImageUsageFlags     usage,
    VkFilter              filter,
    VkSamplerAddressMode  address_mode,
    VkSampleCountFlagBits samples,
    bool                  anisotropic,
    bool                  mipmap) :
    Image(
        VkExtent3D{bitmap->getWidth(), bitmap->getHeight(), 1u},
        format,
        usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        1,
        1,
        layout,
        filter,
        address_mode,
        samples),
    m_anisotropic(anisotropic),
    m_mipmap(mipmap),
    m_bytes_per_pixel(bitmap->getBytesPerPixel())
{
	load(std::move(bitmap));
}

ref<Image2D> Image2D::create(const std::string &path, VkFilter filter, VkSamplerAddressMode address_mode, bool mipmap, bool anisotropic)
{
	auto bitmap = Bitmap::create(path);

	if (!bitmap)
	{
		return nullptr;
	}

	return createRef<Image2D>(
	    std::move(bitmap),
	    bitmap->getBytesPerPixel() == 4 ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R32G32B32A32_SFLOAT,
	    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
	    filter,
	    address_mode,
	    VK_SAMPLE_COUNT_1_BIT,
	    anisotropic,
	    mipmap);
}

void Image2D::load(const scope<Bitmap> &bitmap)
{
	if (bitmap)
	{
		m_extent = {bitmap->getWidth(), bitmap->getHeight(), 1u};
	}

	if (m_extent.width == 0 || m_extent.height == 0)
	{
		return;
	}

	m_mip_levels = m_mipmap ? getMipLevels(m_extent) : 1;

	createImage(m_image, m_allocation, m_extent, VK_IMAGE_TYPE_2D, m_format, m_mip_levels, m_array_layers, m_samples, VK_IMAGE_TILING_OPTIMAL, m_usage, VMA_MEMORY_USAGE_GPU_ONLY);
	createImageSampler(m_sampler, m_filter, m_address_mode, m_anisotropic, m_mip_levels);
	createImageView(m_image, m_view, VK_IMAGE_VIEW_TYPE_2D, m_format, m_mip_levels, 0, m_array_layers, 0, VK_IMAGE_ASPECT_COLOR_BIT);

	if (bitmap || m_mipmap)
	{
		transitionImageLayout(m_image, m_format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, m_mip_levels, 0, m_array_layers, 0);
	}

	if (bitmap)
	{
		// Staging
		//bitmap

		Buffer staging_buffer(bitmap->getLength(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		uint8_t *data;
		staging_buffer.map(reinterpret_cast<void **>(&data));
		std::memcpy(data, bitmap->getData().get(), staging_buffer.getSize());
		staging_buffer.unmap();

		copyBufferToImage(staging_buffer.getBuffer(), m_image, m_extent, m_array_layers, 0);
	}

	if (m_mipmap)
	{
		createMipmaps(m_image, m_extent, m_format, m_layout, m_mip_levels, 0, m_array_layers);
	}
	else if (bitmap)
	{
		transitionImageLayout(m_image, m_format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, m_layout, VK_IMAGE_ASPECT_COLOR_BIT, m_mip_levels, 0, m_array_layers, 0);
	}
	else
	{
		transitionImageLayout(m_image, m_format, VK_IMAGE_LAYOUT_UNDEFINED, m_layout, VK_IMAGE_ASPECT_COLOR_BIT, m_mip_levels, 0, m_array_layers, 0);
	}

	updateDescriptor();
}
}        // namespace Ilum