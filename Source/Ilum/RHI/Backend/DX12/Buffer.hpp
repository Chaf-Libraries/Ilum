#pragma once

#include "RHI/RHIBuffer.hpp"

#include <D3D12MemAlloc.h>

#include <d3d12.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

namespace Ilum::DX12
{
struct BufferState
{
	D3D12_RESOURCE_STATES state;

	inline bool operator==(const BufferState &other)
	{
		return state == other.state;
	}

	static BufferState Create(RHIBufferState state);
};

class Buffer : public RHIBuffer
{
  public:
	Buffer(RHIDevice *device, const BufferDesc &desc);
	virtual ~Buffer() override;

	virtual void *Map() override;
	virtual void  Unmap() override;

	ID3D12Resource* GetHandle();

  private:
	ComPtr<ID3D12Resource> m_handle     = nullptr;
	D3D12MA::Allocation   *m_allocation = nullptr;

	void *m_mapped = nullptr;
};
}        // namespace Ilum::DX12