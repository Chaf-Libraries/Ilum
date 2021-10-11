#pragma once

#include "Utils/PCH.hpp"
#include "Resource/Bitmap/Bitmap.hpp"
#include "Resource/IResource.hpp"

#include "Image.hpp"

namespace Ilum
{
class Image2D : public Image, public IResource<Image2D>
{
  public:
	// Create a 2D texture without any data
	Image2D(
	    const uint32_t       width,
	    const uint32_t       height,
	    VkFormat             format       = VK_FORMAT_R8G8B8A8_UNORM,
	    VkImageLayout        layout       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	    VkImageUsageFlags    usage        = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
	    VkFilter             filter       = VK_FILTER_LINEAR,
	    VkSamplerAddressMode address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
	    VkSampleCountFlagBits samples      = VK_SAMPLE_COUNT_1_BIT,
	    bool                 anisotropic  = false,
	    bool                 mipmap       = false);

	// Create a 2D texture from bitmap
	Image2D(
	    scope<Bitmap> &&            bitmap,
	    VkFormat             format       = VK_FORMAT_R8G8B8A8_UNORM,
	    VkImageLayout        layout       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	    VkImageUsageFlags    usage        = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
	    VkFilter             filter       = VK_FILTER_LINEAR,
	    VkSamplerAddressMode address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
	    VkSampleCountFlagBits samples      = VK_SAMPLE_COUNT_1_BIT,
	    bool                 anisotropic  = false,
	    bool                 mipmap       = false);

	~Image2D() = default;

  public:
	// Create a 2D texture from file for sampling
	static ref<Image2D> create(
	    const std::string &  path,
	    VkFilter             filter       = VK_FILTER_LINEAR,
	    VkSamplerAddressMode address_mode = VK_SAMPLER_ADDRESS_MODE_REPEAT,
	    bool                 mipmap       = true,
	    bool                 anisotropic  = false);

  private:
	void load(const scope<Bitmap> &bitmap = nullptr);

  private:
	const std::string m_path = "";

	bool m_anisotropic = false;
	bool m_mipmap      = true;
	uint32_t m_bytes_per_pixel = 0;
};
}        // namespace Ilum