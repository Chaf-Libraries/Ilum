#include "RHIAccelerationStructure.hpp"
#include "RHIDevice.hpp"

#include "Backend/Vulkan/AccelerationStructure.hpp"

namespace Ilum
{
RHIAccelerationStructure::RHIAccelerationStructure(RHIDevice *device):
    p_device(device)
{
}

std::unique_ptr<RHIAccelerationStructure> RHIAccelerationStructure::Create(RHIDevice *rhi_device)
{
	switch (rhi_device->GetBackend())
	{
		case RHIBackend::Vulkan:
			return std::make_unique<Vulkan::AccelerationStructure>(rhi_device);
		default:
			break;
	}
	return nullptr;
}
}        // namespace Ilum