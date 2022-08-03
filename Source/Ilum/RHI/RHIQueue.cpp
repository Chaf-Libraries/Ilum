#include "RHIQueue.hpp"

#ifdef RHI_BACKEND_VULKAN
#	include "Backend/Vulkan/Queue.hpp"
#elif defined RHI_BACKEND_DX12
#	include "Backend/DX12/Queue.hpp"
#endif        // RHI_BACKEND

namespace Ilum
{
RHIQueue::RHIQueue(RHIDevice *device, RHIQueueFamily family, uint32_t queue_index) :
    p_device(device), m_family(family), m_queue_index(queue_index)
{
}

std::unique_ptr<RHIQueue> RHIQueue::Create(RHIDevice *device, RHIQueueFamily family, uint32_t queue_index)
{
#ifdef RHI_BACKEND_VULKAN
	return std::make_unique<Vulkan::Queue>(device, family, queue_index);
#elif defined RHI_BACKEND_DX12
	return std::make_unique<DX12::Queue>(device, family, queue_index);
#else
	return nullptr;
#endif        // RHI_BACKEND
}
}        // namespace Ilum