#include "RHICommond.hpp"

#include "Vulkan/Commond.hpp"

namespace Ilum
{
RHICommond::RHICommond(RHIDevice *device, RHIQueueFamily family):
    p_device(device), m_family(family)
{
}

std::unique_ptr<RHICommond> RHICommond::Create(RHIDevice *device, RHIQueueFamily family)
{
	return std::make_unique<Vulkan::Commond>(device, family);
}
}        // namespace Ilum