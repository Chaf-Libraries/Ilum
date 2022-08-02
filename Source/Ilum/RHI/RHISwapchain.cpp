#include "RHISwapchain.hpp"
#include "RHIDevice.hpp"

#ifdef RHI_BACKEND_VULKAN
#	include "Vulkan/Swapchain.hpp"
#elif defined RHI_BACKEND_DX12
#	include "DX12/Swapchain.hpp"
#endif

namespace Ilum
{
RHISwapchain::RHISwapchain(RHIDevice *device, Window *window) :
    p_device(device), p_window(window)
{
}

std::unique_ptr<RHISwapchain> RHISwapchain::Create(RHIDevice *device, Window *window)
{
#ifdef RHI_BACKEND_VULKAN
	return std::make_unique<Vulkan::Swapchain>(device, window);
#elif defined RHI_BACKEND_DX12
	return std::make_unique<DX12::Swapchain>(device, window);
#else
	return nullptr;
#endif
}

}        // namespace Ilum