#include "RHIDevice.hpp"

#ifdef RHI_BACKEND_VULKAN
#include "Backend/Vulkan/Device.hpp"
#elif defined RHI_BACKEND_DX12
#	include "Backend/DX12/Device.hpp"
#endif

namespace Ilum
{
std::unique_ptr<RHIDevice> RHIDevice::Create()
{
#ifdef RHI_BACKEND_VULKAN
	return std::make_unique<Vulkan::Device>();
#elif defined RHI_BACKEND_DX12
	return std::make_unique<DX12::Device>();
#endif        // RHI_BACKEND_VULKAN
	return nullptr;
}
}        // namespace Ilum