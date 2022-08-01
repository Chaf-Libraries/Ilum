#include "RHISynchronization.hpp"

#include "Vulkan/Synchronization.hpp"

namespace Ilum
{
RHIFence::RHIFence(RHIDevice *device) :
    p_device(device)
{
}

std::unique_ptr<RHIFence> RHIFence::Create(RHIDevice *device)
{
	return std::make_unique<Vulkan::Fence>(device);
}

RHISemaphore::RHISemaphore(RHIDevice *device) :
    p_device(device)
{
}

std::unique_ptr<RHISemaphore> RHISemaphore::Create(RHIDevice *device)
{
	return std::make_unique<Vulkan::Semaphore>(device);
}
}        // namespace Ilum