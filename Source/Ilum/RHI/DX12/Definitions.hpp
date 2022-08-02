#pragma once

#include "RHI/RHIDefinitions.hpp"

#include <d3d12.h>

#include <unordered_map>

namespace Ilum::DX12
{
inline static std::unordered_map<RHIFormat, DXGI_FORMAT> ToDX12Format = {
    {RHIFormat::R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM},
    {RHIFormat::R16G16B16A16_SFLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT},
    {RHIFormat::R32G32B32A32_SFLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT},
    {RHIFormat::D32_SFLOAT, DXGI_FORMAT_D32_FLOAT},
    {RHIFormat::D24_UNORM_S8_UINT, DXGI_FORMAT_D24_UNORM_S8_UINT},
};

inline static D3D12_RESOURCE_FLAGS ToDX12ResourceFlags(RHITextureUsage usage)
{
	D3D12_RESOURCE_FLAGS dx_flsgs = D3D12_RESOURCE_FLAG_NONE;

	if (!(usage & RHITextureUsage::ShaderResource))
	{
		dx_flsgs |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
	}
	if (usage & RHITextureUsage::UnorderedAccess)
	{
		dx_flsgs |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
	}

	return dx_flsgs;
}

inline static D3D12_HEAP_TYPE ToDX12HeapType(RHIMemoryUsage usage)
{
	switch (usage)
	{
		case RHIMemoryUsage::GPU_Only:
			return D3D12_HEAP_TYPE_DEFAULT;
		case RHIMemoryUsage::CPU_TO_GPU:
			return D3D12_HEAP_TYPE_UPLOAD;
		case RHIMemoryUsage::GPU_TO_CPU:
			return D3D12_HEAP_TYPE_READBACK;
		default:
			break;
	}
	return D3D12_HEAP_TYPE_DEFAULT;
}

}        // namespace Ilum::DX12