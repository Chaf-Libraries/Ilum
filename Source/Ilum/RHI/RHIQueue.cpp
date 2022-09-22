#include "RHIQueue.hpp"
#include "RHIDevice.hpp"

#include "Backend/Vulkan/Queue.hpp"
#include "Backend/DX12/Queue.hpp"
#include "Backend/CUDA/Queue.hpp"

namespace Ilum
{
RHIQueue::RHIQueue(RHIDevice *device)
{
}

std::unique_ptr<RHIQueue> RHIQueue::Create(RHIDevice *device)
{
	switch (device->GetBackend())
	{
		case RHIBackend::Vulkan:
			return std::make_unique<Vulkan::Queue>(device);
		case RHIBackend::DX12:
			//return std::make_unique<DX12::Queue>(device, family, queue_index);
		case RHIBackend::CUDA:
			//return std::make_unique<CUDA::Queue>(device, family, queue_index);
		default:
			break;
	}
	return nullptr;
}
}        // namespace Ilum