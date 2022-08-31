#include "Swapchain.hpp"

namespace Ilum::CUDA
{
Swapchain::Swapchain(RHIDevice *device, uint32_t width, uint32_t height, bool vsync) :
    RHISwapchain(device, width, height, vsync)
{
}

uint32_t Swapchain::GetTextureCount()
{
	return 1;
}

void Swapchain::AcquireNextTexture(RHISemaphore *signal_semaphore, RHIFence *signal_fence)
{
}

RHITexture *Swapchain::GetCurrentTexture()
{
	return nullptr;
}

uint32_t Swapchain::GetCurrentFrameIndex()
{
	return 0;
}

bool Swapchain::Present(RHISemaphore *semaphore)
{
    return true;
}

void Swapchain::Resize(uint32_t width, uint32_t height)
{
}
}        // namespace Ilum::CUDA