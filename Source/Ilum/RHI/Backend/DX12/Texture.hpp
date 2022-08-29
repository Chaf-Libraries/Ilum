#pragma once

#include "RHI/RHITexture.hpp"

#include <D3D12MemAlloc.h>

#include <d3d12.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

namespace Ilum::DX12
{
struct TextureState
{
	D3D12_RESOURCE_STATES state;

	inline bool operator==(const TextureState &other)
	{
		return state == other.state;
	}

	static TextureState Create(RHIResourceState state);
};

class Texture : public RHITexture
{
  public:
	Texture(RHIDevice *device, const TextureDesc &desc);
	Texture(RHIDevice *device, const TextureDesc &desc, ComPtr<ID3D12Resource> &&texture);

	virtual ~Texture() override;

	ID3D12Resource* GetHandle();

  private:
	ComPtr<ID3D12Resource> m_handle     = nullptr;
	D3D12MA::Allocation   *m_allocation = nullptr;
};
}        // namespace Ilum::DX12