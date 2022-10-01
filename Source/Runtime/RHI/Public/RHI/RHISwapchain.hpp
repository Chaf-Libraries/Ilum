#pragma once

#include <Core/Window.hpp>

namespace Ilum
{
class RHISemaphore;
class RHIFence;
class RHIDevice;
class RHITexture;

class RHISwapchain
{
  public:
	RHISwapchain(RHIDevice *device, uint32_t width, uint32_t height, bool vsync = false);

	virtual ~RHISwapchain() = default;

	uint32_t GetWidth() const;

	uint32_t GetHeight() const;

	static std::unique_ptr<RHISwapchain> Create(RHIDevice *device, void *window_handle, uint32_t width, uint32_t height, bool sync);

	virtual uint32_t GetTextureCount() = 0;

	virtual void AcquireNextTexture(RHISemaphore *signal_semaphore, RHIFence* signal_fence) = 0;

	virtual RHITexture *GetCurrentTexture() = 0;

	virtual uint32_t GetCurrentFrameIndex() = 0;

	virtual bool Present(RHISemaphore *semaphore) = 0;

	virtual void Resize(uint32_t width, uint32_t height) = 0;

  protected:
	RHIDevice *p_device = nullptr;
	uint32_t   m_width  = 0;
	uint32_t   m_height = 0;
	bool       m_vsync  = false;
};
}        // namespace Ilum