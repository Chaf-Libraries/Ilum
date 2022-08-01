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
	RHISwapchain(RHIDevice *device, Window *window);
	virtual ~RHISwapchain() = default;

	static std::unique_ptr<RHISwapchain> Create(RHIDevice *device, Window *window);

	virtual uint32_t GetTextureCount() = 0;

	virtual void AcquireNextTexture(RHISemaphore *semaphore, RHIFence* fence) = 0;

	virtual RHITexture *GetCurrentTexture() = 0;

	virtual uint32_t GetCurrentFrameIndex() = 0;

	virtual void Present(RHISemaphore *semaphore) = 0;

  protected:
	RHIDevice *p_device = nullptr;
	Window    *p_window = nullptr;
};
}        // namespace Ilum