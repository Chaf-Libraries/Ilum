#pragma once

#include "RHI/RHIQueue.hpp"

#include <d3d12.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

namespace Ilum::DX12
{
class Queue : public RHIQueue
{
  public:
	Queue(RHIDevice *device, RHIQueueFamily family, uint32_t queue_index = 0);

	virtual ~Queue() override = default;

	virtual void Wait() override;

	virtual void Submit(const std::vector<RHICommand *> &cmds, const std::vector<RHISemaphore *> &signal_semaphores = {}, const std::vector<RHISemaphore *> &wait_semaphores = {}) override;

	virtual void Execute(RHIFence *fence = nullptr) override;

	virtual bool Empty() override;

  private:
	ComPtr<ID3D12CommandQueue> m_handle = nullptr;
};
}        // namespace Ilum::DX12