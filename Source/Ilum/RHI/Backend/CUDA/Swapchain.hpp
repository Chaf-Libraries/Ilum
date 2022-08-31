#pragma once

#include "RHI/RHISwapchain.hpp"

namespace Ilum::CUDA
{
class Swapchain : public RHISwapchain
{
  public:
	Swapchain(RHIDevice *device, uint32_t width, uint32_t height, bool vsync = false);

	virtual ~Swapchain() = default;

	virtual uint32_t GetTextureCount() override;

	virtual void AcquireNextTexture(RHISemaphore *signal_semaphore, RHIFence *signal_fence) override;

	virtual RHITexture *GetCurrentTexture() override;

	virtual uint32_t GetCurrentFrameIndex() override;

	virtual bool Present(RHISemaphore *semaphore) override;

	virtual void Resize(uint32_t width, uint32_t height) override;
};
}        // namespace Ilum::CUDA