#include "Device.hpp"

#include <Windows.h>
#include <comdef.h>

#include <dxgi1_6.h>

namespace Ilum::DX12
{
inline void GetHardwareAdapter(IDXGIFactory1 *pFactory, IDXGIAdapter1 **ppAdapter)
{
	*ppAdapter = nullptr;

	ComPtr<IDXGIAdapter1> adapter;

	ComPtr<IDXGIFactory6> factory6;
	if (SUCCEEDED(pFactory->QueryInterface(IID_PPV_ARGS(&factory6))))
	{
		for (
		    UINT adapterIndex = 0;
		    SUCCEEDED(factory6->EnumAdapterByGpuPreference(
		        adapterIndex,
		        DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
		        IID_PPV_ARGS(&adapter)));
		    ++adapterIndex)
		{
			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);

			std::stringstream ss;
			ss << "\nFound physical device [" << desc.DeviceId << "]\n";
			ss << "Name: " << _bstr_t(desc.Description) << "\n";
			ss << "Vendor: ";
			switch (desc.VendorId)
			{
				case 0x8086:
					ss << "Intel\n";
					break;
				case 0x10DE:
					ss << "Nvidia\n";
					break;
				case 0x1002:
					ss << "AMD\n";
					break;
				default:
					ss << desc.VendorId << "\n";
			}

			LOG_INFO("{}", ss.str());

			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
			{
				continue;
			}

			if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
			{
				break;
			}
		}
	}

	if (adapter.Get() == nullptr)
	{
		for (UINT adapterIndex = 0; SUCCEEDED(pFactory->EnumAdapters1(adapterIndex, &adapter)); ++adapterIndex)
		{
			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);

			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
			{
				continue;
			}

			if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
			{
				break;
			}
		}
	}

	*ppAdapter = adapter.Detach();
}

Device::Device()
{
	uint32_t dxgi_factory_flags = 0;

#ifdef _DEBUG
	ComPtr<ID3D12Debug> debug_controller;
	D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller));
	debug_controller->EnableDebugLayer();
	dxgi_factory_flags |= DXGI_CREATE_FACTORY_DEBUG;
#endif        // _DEBUG

	CreateDXGIFactory2(dxgi_factory_flags, IID_PPV_ARGS(&m_factory));

	ComPtr<IDXGIAdapter1> hardware_adapter;
	GetHardwareAdapter(m_factory.Get(), &hardware_adapter);

	D3D12CreateDevice(hardware_adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_handle));

	D3D12MA::ALLOCATOR_DESC allocator_desc = {};
	allocator_desc.Flags                   = D3D12MA::ALLOCATOR_FLAG_NONE;
	allocator_desc.pDevice                 = m_handle.Get();
	allocator_desc.pAdapter                = hardware_adapter.Get();

	D3D12MA::CreateAllocator(&allocator_desc, &m_allocator);

	// Check feature support
	{
		D3D12_FEATURE_DATA_D3D12_OPTIONS7 feature = {};
		m_handle->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &feature, sizeof(feature));
		m_feature_support[RHIFeature::MeshShading] = feature.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED;
	}
	{
		D3D12_FEATURE_DATA_D3D12_OPTIONS5 feature = {};
		m_handle->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &feature, sizeof(feature));
		m_feature_support[RHIFeature::RayTracing] = feature.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
	}
	{
		D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT feature = {};
		m_handle->CheckFeatureSupport(D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &feature, sizeof(feature));
		m_feature_support[RHIFeature::BufferDeviceAddress] = feature.MaxGPUVirtualAddressBitsPerProcess > 0 || feature.MaxGPUVirtualAddressBitsPerResource > 0;
	}
	{
		// TODO: I don't know where to check bindless, set to true for now
		m_feature_support[RHIFeature::Bindless] = true;
	}

	LOG_INFO("Feature RayTracing Support: {}", m_feature_support[RHIFeature::RayTracing]);
	LOG_INFO("Feature MeshShading Support: {}", m_feature_support[RHIFeature::MeshShading]);
	LOG_INFO("Feature Buffer Device Address Support: {}", m_feature_support[RHIFeature::BufferDeviceAddress]);
	LOG_INFO("Feature Bindless Support: {}", m_feature_support[RHIFeature::Bindless]);
}

Device::~Device()
{
	m_allocator->Release();
	m_allocator = nullptr;
}

void Device::WaitIdle()
{
}

bool Device::IsFeatureSupport(RHIFeature feature)
{
	return m_feature_support[feature];
}

ID3D12Device*Device::GetHandle()
{
	return m_handle.Get();
}

IDXGIFactory4 *Device::GetFactory()
{
	return m_factory.Get();
}

D3D12MA::Allocator *Device::GetAllocator()
{
	return m_allocator;
}

}        // namespace Ilum::DX12