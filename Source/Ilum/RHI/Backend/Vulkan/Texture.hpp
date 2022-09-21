#pragma once

#include "RHI/RHITexture.hpp"

#include <volk.h>

#include <vk_mem_alloc.h>

namespace Ilum::Vulkan
{
struct TextureState
{
	VkImageLayout        layout      = VK_IMAGE_LAYOUT_UNDEFINED;
	VkAccessFlags        access_mask = VK_ACCESS_NONE;
	VkPipelineStageFlags stage       = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;

	inline bool operator==(const TextureState &other)
	{
		return layout == other.layout &&
		       access_mask == other.access_mask &&
		       stage == other.stage;
	}

	static TextureState Create(RHIResourceState state);
};

class Texture : public RHITexture
{
  public:
	Texture(RHIDevice *device, const TextureDesc &desc);

	Texture(RHIDevice *device, const TextureDesc &desc, VkImage image, bool is_swapchain_buffer = true);

	virtual std::unique_ptr<RHITexture> Alias(const TextureDesc &desc) override;

	virtual ~Texture() override;

	VkImage GetHandle() const;

	VkImageView GetView(const TextureRange &range) const;

  private:
	VkImage       m_handle     = VK_NULL_HANDLE;
	VmaAllocation m_allocation = VK_NULL_HANDLE;
	VkDeviceMemory m_memory     = VK_NULL_HANDLE;

	size_t m_memory_size = 0;

	bool m_is_swapchain_buffer = false;

	mutable std::unordered_map<size_t, VkImageView> m_view_cache;
};
}        // namespace Ilum::Vulkan