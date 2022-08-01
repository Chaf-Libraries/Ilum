#include "RHIQueue.hpp"

#include "Vulkan/Queue.hpp"

namespace Ilum
{
RHIQueue::RHIQueue(RHIDevice *device, RHIQueueFamily family, uint32_t queue_index) :
    p_device(device), m_family(family), m_queue_index(queue_index)
{
}

std::unique_ptr<RHIQueue> RHIQueue::Create(RHIDevice *device, RHIQueueFamily family, uint32_t queue_index)
{
	return std::make_unique<Vulkan::Queue>(device, family, queue_index);
}
}        // namespace Ilum