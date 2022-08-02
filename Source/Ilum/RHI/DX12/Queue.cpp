#pragma once

#include "Queue.hpp"
#include "Device.hpp"

namespace Ilum::DX12
{
Queue::Queue(RHIDevice *device, RHIQueueFamily family, uint32_t queue_index):
    RHIQueue(device,family, queue_index)
{
	D3D12_COMMAND_QUEUE_DESC desc = {};
	desc.Flags                    = D3D12_COMMAND_QUEUE_FLAG_NONE;
	switch (family)
	{
		case RHIQueueFamily::Graphics:
			desc.Type = D3D12_COMMAND_LIST_TYPE_BUNDLE;
			break;
		case RHIQueueFamily::Compute:
			desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
			break;
		case RHIQueueFamily::Transfer:
			desc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
			break;
		default:
			break;
	}
	static_cast<Device *>(p_device)->GetHandle()->CreateCommandQueue(&desc, IID_PPV_ARGS(&m_handle));
}

void Queue::Submit(const std::vector<RHICommand *> &cmds, const std::vector<RHISemaphore *> &signal_semaphores, const std::vector<RHISemaphore *> &wait_semaphores)
{
}

void Queue::Execute(RHIFence *fence)
{
}
}        // namespace Ilum::DX12