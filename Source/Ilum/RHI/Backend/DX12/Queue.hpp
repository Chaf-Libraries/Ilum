#pragma once

#include "RHI/RHIQueue.hpp"

#include <d3d12.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

namespace Ilum::DX12
{
class Device;

class Queue : public RHIQueue
{
  public:
	Queue(RHIDevice *device);

	virtual ~Queue() override = default;

	virtual void Execute(RHIQueueFamily family, const std::vector<SubmitInfo> &submit_infos, RHIFence *fence) override;

	// Immediate execution
	virtual void Execute(RHICommand *cmd_buffer) override;

	virtual void Wait() override;

  private:
	Device *p_device;

	ComPtr<ID3D12CommandQueue> m_handle = nullptr;
};
}        // namespace Ilum::DX12