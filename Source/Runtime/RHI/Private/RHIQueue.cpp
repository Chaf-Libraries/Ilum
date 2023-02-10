#include "RHIQueue.hpp"
#include "RHIDevice.hpp"

#include <Core/Plugin.hpp>

namespace Ilum
{
RHIQueue::RHIQueue(RHIDevice *device)
{
}

std::unique_ptr<RHIQueue> RHIQueue::Create(RHIDevice *device)
{
	return std::unique_ptr<RHIQueue>(std::move(PluginManager::GetInstance().Call<RHIQueue *>(fmt::format("shared/RHI/RHI.{}.dll", device->GetBackend()), "CreateQueue", device)));
}
}        // namespace Ilum