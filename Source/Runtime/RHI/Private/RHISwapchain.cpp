#include "RHISwapchain.hpp"
#include "RHIDevice.hpp"

#include <Core/Plugin.hpp>

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
	return std::unique_ptr<RHISwapchain>(std::move(PluginManager::GetInstance().Call<RHISwapchain *>(fmt::format("shared/RHI/RHI.{}.dll", device->GetBackend()), "CreateSwapchain", device, window_handle, width, height, vsync)));
}

}        // namespace Ilum