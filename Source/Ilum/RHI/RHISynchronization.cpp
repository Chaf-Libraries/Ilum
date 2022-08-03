#include "RHISynchronization.hpp"

#ifdef RHI_BACKEND_VULKAN
#	include "Backend/Vulkan/Synchronization.hpp"
#elif defined RHI_BACKEND_DX12
#	include "Backend/DX12/Synchronization.hpp"
#endif        // RHI_BACKEND

namespace Ilum
{
RHIFence::RHIFence(RHIDevice *device) :
    p_device(device)
{
}

std::unique_ptr<RHIFence> RHIFence::Create(RHIDevice *device)
{
#ifdef RHI_BACKEND_VULKAN
	return std::make_unique<Vulkan::Fence>(device);
#elif defined RHI_BACKEND_DX12
	return std::make_unique<DX12::Fence>(device);
#else
	return nullptr;
#endif        // RHI_BACKEND
}

RHISemaphore::RHISemaphore(RHIDevice *device) :
    p_device(device)
{
}

std::unique_ptr<RHISemaphore> RHISemaphore::Create(RHIDevice *device)
{
#ifdef RHI_BACKEND_VULKAN
	return std::make_unique<Vulkan::Semaphore>(device);
#elif defined RHI_BACKEND_DX12
	return std::make_unique<DX12::Semaphore>(device);
#else
	return nullptr;
#endif        // RHI_BACKEND
}
}        // namespace Ilum