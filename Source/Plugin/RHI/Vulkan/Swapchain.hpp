#pragma once

#include "Fwd.hpp"

namespace Ilum::Vulkan
{
class Texture;
class Queue;
class Device;

class Swapchain : public RHISwapchain
{
  public:
	Swapchain(Device *device, void *window_handle, uint32_t width, uint32_t height, bool vsync);

	virtual ~Swapchain() override;

	virtual uint32_t GetTextureCount() override;

	virtual void AcquireNextTexture(RHISemaphore *signal_semaphore, RHIFence *signal_fence) override;

	virtual RHITexture *GetCurrentTexture() override;

	virtual uint32_t GetCurrentFrameIndex() override;

	virtual bool Present(RHISemaphore *semaphore) override;

	virtual void Resize(uint32_t width, uint32_t height, bool vsync) override;

  private:
	VkSurfaceKHR   m_surface   = VK_NULL_HANDLE;
	VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;

	VkQueue m_present_queue = VK_NULL_HANDLE;
	uint32_t m_present_family = 0;

	std::vector<std::unique_ptr<Texture>> m_textures;

	uint32_t                 m_image_count = 0;
	VkSurfaceFormatKHR       m_surface_format;
	VkSurfaceCapabilitiesKHR m_capabilities;

	uint32_t m_frame_index = 0;
};
}        // namespace Ilum::Vulkan