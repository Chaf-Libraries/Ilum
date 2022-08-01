#include "RHISwapchain.hpp"
#include "RHIDevice.hpp"

#include "Vulkan/Swapchain.hpp"

namespace Ilum
{
RHISwapchain::RHISwapchain(RHIDevice *device, Window *window):
    p_device(device), p_window(window)
{
}

std::unique_ptr<RHISwapchain> RHISwapchain::Create(RHIDevice *device, Window *window)
{
	return std::make_unique<Vulkan::Swapchain>(device, window);
}

}        // namespace Ilum