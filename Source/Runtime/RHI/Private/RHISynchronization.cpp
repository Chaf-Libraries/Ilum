#include "RHISynchronization.hpp"
#include "RHIDevice.hpp"

#include <Core/Plugin.hpp>

namespace Ilum
{
RHIFence::RHIFence(RHIDevice *device) :
    p_device(device)
{
}

std::unique_ptr<RHIFence> RHIFence::Create(RHIDevice *device)
{
	return std::unique_ptr<RHIFence>(std::move(PluginManager::GetInstance().Call<RHIFence *>(fmt::format("RHI.{}.dll", device->GetBackend()), "CreateFence", device)));
}

RHISemaphore::RHISemaphore(RHIDevice *device) :
    p_device(device)
{
}

std::unique_ptr<RHISemaphore> RHISemaphore::Create(RHIDevice *device)
{
	return std::unique_ptr<RHISemaphore>(std::move(PluginManager::GetInstance().Call<RHISemaphore *>(fmt::format("RHI.{}.dll", device->GetBackend()), "CreateSemaphore", device)));
}
}        // namespace Ilum