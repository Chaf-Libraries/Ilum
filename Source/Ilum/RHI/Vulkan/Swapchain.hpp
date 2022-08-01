#pragma once

#include "RHI/RHISwapchain.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
class Texture;
class Queue;

class Swapchain : public RHISwapchain
{
  public:
	Swapchain(RHIDevice *device, Window *window);
	virtual ~Swapchain() override;

	virtual uint32_t GetTextureCount() override;

	virtual void AcquireNextTexture(RHISemaphore *semaphore, RHIFence *fence) override;

	virtual RHITexture *GetCurrentTexture() override;

	virtual uint32_t GetCurrentFrameIndex() override;

	virtual void Present(RHISemaphore *semaphore) override;

  private:
	void CreateSwapchain(const VkExtent2D &extent);

  private:
	VkSurfaceKHR   m_surface   = VK_NULL_HANDLE;
	VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;

	std::unique_ptr<Queue> m_present_queue = nullptr;

	std::vector<std::unique_ptr<Texture>> m_textures;

	uint32_t                 m_image_count = 0;
	VkSurfaceFormatKHR       m_surface_format;
	VkSurfaceCapabilitiesKHR m_capabilities;
	VkPresentModeKHR         m_present_mode;

	uint32_t m_frame_index = 0;
};
}        // namespace Ilum::Vulkan