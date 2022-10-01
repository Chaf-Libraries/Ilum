#pragma once

#include "Queue.hpp"
#include "Device.hpp"
#include "Synchronization.hpp"

namespace Ilum::DX12
{
Queue::Queue(RHIDevice *device) :
    RHIQueue(device)
{
	D3D12_COMMAND_QUEUE_DESC desc = {};
	desc.Flags                    = D3D12_COMMAND_QUEUE_FLAG_NONE;
	desc.Type                     = D3D12_COMMAND_LIST_TYPE_DIRECT;
	p_device->GetHandle()->CreateCommandQueue(&desc, IID_PPV_ARGS(&m_handle));
}

void Queue::Execute(RHIQueueFamily family, const std::vector<SubmitInfo> &submit_infos, RHIFence *fence)
{
}

// Immediate execution
void Queue::Execute(RHICommand *cmd_buffer)
{
}

void Queue::Wait()
{
}

// void Queue::Wait()
//{
// }
//
// void Queue::Submit(const std::vector<RHICommand *> &cmds, const std::vector<RHISemaphore *> &signal_semaphores, const std::vector<RHISemaphore *> &wait_semaphores)
//{
// }
//
// void Queue::Execute(RHIFence *fence)
//{
//	// m_handle->ExecuteCommandLists(0, nullptr);
//	if (fence)
//	{
//		Fence *dx_fence = static_cast<Fence *>(fence);
//		// uint64_t fence_val = dx_fence->GetValue() + 1;
//		// m_handle->Signal(dx_fence->GetHandle().Get(), fence_val);
//	}
// }
//
// bool Queue::Empty()
//{
//	return true;
// }
}        // namespace Ilum::DX12