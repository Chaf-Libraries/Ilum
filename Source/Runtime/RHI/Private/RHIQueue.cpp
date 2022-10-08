#include "RHIQueue.hpp"
#include "RHIDevice.hpp"

#include "Backend/DX12/Queue.hpp"
#include "Backend/Vulkan/Queue.hpp"
#ifdef CUDA_ENABLE
#	include "Backend/CUDA/Queue.hpp"
#endif        // CUDA_ENABLE

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
			// return std::make_unique<DX12::Queue>(device, family, queue_index);
#ifdef CUDA_ENABLE
		case RHIBackend::CUDA:
			return std::make_unique<CUDA::Queue>(device);
#endif        // CUDA_ENABLE
		default:
			break;
	}
	return nullptr;
}
}        // namespace Ilum