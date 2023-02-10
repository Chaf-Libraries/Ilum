#include "RHIAccelerationStructure.hpp"
#include "RHIDevice.hpp"

#include <Core/Plugin.hpp>

namespace Ilum
{
RHIAccelerationStructure::RHIAccelerationStructure(RHIDevice *device):
    p_device(device)
{
}

std::unique_ptr<RHIAccelerationStructure> RHIAccelerationStructure::Create(RHIDevice *rhi_device)
{
	return std::unique_ptr<RHIAccelerationStructure>(std::move(PluginManager::GetInstance().Call<RHIAccelerationStructure *>(fmt::format("RHI.{}.dll", rhi_device->GetBackend()), "CreateAccelerationStructure", rhi_device)));
}
}        // namespace Ilum