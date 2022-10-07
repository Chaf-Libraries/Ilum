#include "RHISwapchain.hpp"
#include "RHIDevice.hpp"

#include "Backend/Vulkan/Swapchain.hpp"
#include "Backend/DX12/Swapchain.hpp"

namespace Ilum
{
RHISwapchain::RHISwapchain(RHIDevice *device, uint32_t width, uint32_t height, bool vsync) :
    p_device(device), m_width(width), m_height(height), m_vsync(vsync)
{
}

bool RHISwapchain::GetVsync() const
{
	return m_vsync;
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
	switch (device->GetBackend())
	{
		case RHIBackend::Vulkan:
			return std::make_unique<Vulkan::Swapchain>(device, window_handle, width, height, vsync);
		case RHIBackend::DX12:
			return std::make_unique<DX12::Swapchain>(device, window_handle, width, height, vsync);
		default:
			break;
	}
	return nullptr;
}

}        // namespace Ilum