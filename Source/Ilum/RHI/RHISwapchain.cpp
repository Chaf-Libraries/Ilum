#include "RHISwapchain.hpp"
#include "RHIDevice.hpp"

#ifdef RHI_BACKEND_VULKAN
#	include "Backend/Vulkan/Swapchain.hpp"
#elif defined RHI_BACKEND_DX12
#	include "Backend/DX12/Swapchain.hpp"
#endif

namespace Ilum
{
RHISwapchain::RHISwapchain(RHIDevice *device, uint32_t width, uint32_t height, bool vsync) :
    p_device(device), m_width(width), m_height(height), m_vsync(vsync)
{
}

uint32_t RHISwapchain::GetWidth() const
{
	return m_width;
}

uint32_t RHISwapchain::GetHeight() const
{
	return m_height;
}

std::unique_ptr<RHISwapchain> RHISwapchain::Create(RHIDevice *device, void *window_handle, uint32_t width, uint32_t height, bool vsync)
{
#ifdef RHI_BACKEND_VULKAN
	return std::make_unique<Vulkan::Swapchain>(device, window_handle, width, height, vsync);
#elif defined RHI_BACKEND_DX12
	return std::make_unique<DX12::Swapchain>(device, window_handle, width, height, vsync);
#else
	return nullptr;
#endif
}

}        // namespace Ilum