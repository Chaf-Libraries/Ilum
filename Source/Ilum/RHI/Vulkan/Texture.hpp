#pragma once

#include "RHI/RHITexture.hpp"

#include <volk.h>
#include <vk_mem_alloc.h>

namespace Ilum::Vulkan
{
class Texture : public RHITexture
{
  public:
	Texture(RHIDevice *device, const TextureDesc &desc);
	Texture(RHIDevice *device, const TextureDesc &desc, VkImage image);
	virtual ~Texture() override;

	VkImage GetHandle() const;

	VkImageView GetView(const TextureRange &range) const;

  private:
	VkImage m_handle = VK_NULL_HANDLE;
	VmaAllocation m_allocation = VK_NULL_HANDLE;

	mutable std::unordered_map<size_t, VkImageView> m_view_cache;
};
}        // namespace Ilum::Vulkan