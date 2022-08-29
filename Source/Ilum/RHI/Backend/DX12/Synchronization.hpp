#pragma once

#include "RHI/RHISynchronization.hpp"

#include <d3d12.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

namespace Ilum::DX12
{
class Fence : public RHIFence
{
  public:
	Fence(RHIDevice *device);
	virtual ~Fence() = default;

	virtual void Wait(uint64_t timeout) override;
	virtual void Reset() override;

	ID3D12Fence *GetHandle();
	uint64_t    &GetValue();

  private:
	ComPtr<ID3D12Fence> m_handle      = nullptr;
	uint64_t            m_fence_value = 0;
};

class Semaphore : public RHISemaphore
{
  public:
	Semaphore(RHIDevice *device);
};
}        // namespace Ilum::DX12