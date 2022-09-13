#include "RHISynchronization.hpp"
#include "RHIDevice.hpp"

#include "Backend/Vulkan/Synchronization.hpp"
#include "Backend/DX12/Synchronization.hpp"

namespace Ilum
{
RHIFence::RHIFence(RHIDevice *device) :
    p_device(device)
{
}

std::unique_ptr<RHIFence> RHIFence::Create(RHIDevice *device)
{
	switch (device->GetBackend())
	{
		case RHIBackend::Vulkan:
			return std::make_unique<Vulkan::Fence>(device);
		case RHIBackend::DX12:
			return std::make_unique<DX12::Fence>(device);
		default:
			break;
	}
	return nullptr;
}

RHISemaphore::RHISemaphore(RHIDevice *device) :
    p_device(device)
{
}

std::unique_ptr<RHISemaphore> RHISemaphore::Create(RHIDevice *device)
{
	switch (device->GetBackend())
	{
		case RHIBackend::Vulkan:
			return std::make_unique<Vulkan::Semaphore>(device);
		case RHIBackend::DX12:
			return std::make_unique<DX12::Semaphore>(device);
		default:
			break;
	}
	return nullptr;
}
}        // namespace Ilum