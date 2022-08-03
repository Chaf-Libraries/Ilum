#pragma once

#include "RHI/RHIDevice.hpp"

#include <D3D12MemAlloc.h>

#include <d3d12.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

namespace Ilum::DX12
{
class Device : public RHIDevice
{
  public:
	Device();
	virtual ~Device() override;

	virtual void WaitIdle() override;

	virtual bool IsFeatureSupport(RHIFeature feature) override;

	ComPtr<ID3D12Device> &GetHandle();
	ComPtr<IDXGIFactory4> &GetFactory();
	D3D12MA::Allocator   *GetAllocator();

  private:
	ComPtr<ID3D12Device>                 m_handle    = nullptr;
	ComPtr<IDXGIFactory4>                m_factory   = nullptr;
	D3D12MA::Allocator                  *m_allocator = nullptr;
	std::unordered_map<RHIFeature, bool> m_feature_support;
};
}        // namespace Ilum::DX12