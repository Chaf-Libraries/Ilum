#pragma once

#include "RHI/RHIDefinitions.hpp"

#include <directx/d3d12.h>

#include <unordered_map>

namespace Ilum::DX12
{
inline static std::unordered_map<RHIFormat, DXGI_FORMAT> ToDX12Format = {
    {RHIFormat::Undefined, DXGI_FORMAT_UNKNOWN},
    {RHIFormat::R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM},
    {RHIFormat::R16_UINT, DXGI_FORMAT_R16_UINT},
    {RHIFormat::R16_SINT, DXGI_FORMAT_R16_SINT},
    {RHIFormat::R16_FLOAT, DXGI_FORMAT_R16_FLOAT},
    {RHIFormat::R16G16_UINT, DXGI_FORMAT_R16G16_UINT},
    {RHIFormat::R16G16_SINT, DXGI_FORMAT_R16G16_SINT},
    {RHIFormat::R16G16_FLOAT, DXGI_FORMAT_R16G16_FLOAT},
    {RHIFormat::R16G16B16A16_UINT, DXGI_FORMAT_R16G16B16A16_UINT},
    {RHIFormat::R16G16B16A16_SINT, DXGI_FORMAT_R16G16B16A16_SINT},
    {RHIFormat::R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT},
    {RHIFormat::R32_UINT, DXGI_FORMAT_R32_UINT},
    {RHIFormat::R32_SINT, DXGI_FORMAT_R32_SINT},
    {RHIFormat::R32_FLOAT, DXGI_FORMAT_R32_FLOAT},
    {RHIFormat::R32G32_UINT, DXGI_FORMAT_R32G32_UINT},
    {RHIFormat::R32G32_SINT, DXGI_FORMAT_R32G32_SINT},
    {RHIFormat::R32G32_FLOAT, DXGI_FORMAT_R32G32_FLOAT},
    {RHIFormat::R32G32B32_UINT, DXGI_FORMAT_R32G32B32_UINT},
    {RHIFormat::R32G32B32_SINT, DXGI_FORMAT_R32G32B32_SINT},
    {RHIFormat::R32G32B32_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT},
    {RHIFormat::R32G32B32A32_UINT, DXGI_FORMAT_R32G32B32A32_UINT},
    {RHIFormat::R32G32B32A32_SINT, DXGI_FORMAT_R32G32B32A32_SINT},
    {RHIFormat::R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT},
    {RHIFormat::D32_FLOAT, DXGI_FORMAT_D32_FLOAT},
    {RHIFormat::D24_UNORM_S8_UINT, DXGI_FORMAT_D24_UNORM_S8_UINT},
};

inline static std::unordered_map<RHIAddressMode, D3D12_TEXTURE_ADDRESS_MODE> ToDX12AddressMode = {
    {RHIAddressMode::Repeat, D3D12_TEXTURE_ADDRESS_MODE_WRAP},
    {RHIAddressMode::Mirrored_Repeat, D3D12_TEXTURE_ADDRESS_MODE_MIRROR},
    {RHIAddressMode::Clamp_To_Edge, D3D12_TEXTURE_ADDRESS_MODE_CLAMP},
    {RHIAddressMode::Clamp_To_Border, D3D12_TEXTURE_ADDRESS_MODE_BORDER},
    {RHIAddressMode::Mirror_Clamp_To_Edge, D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE},
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

inline static D3D12_RESOURCE_FLAGS ToDX12ResourceFlags(RHIBufferUsage usage)
{
	D3D12_RESOURCE_FLAGS dx_flsgs = D3D12_RESOURCE_FLAG_NONE;

	if (!(usage & RHIBufferUsage::ShaderResource) && !(usage & RHIBufferUsage::ConstantBuffer))
	{
		dx_flsgs |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
	}
	if (usage & RHIBufferUsage::UnorderedAccess)
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

inline static D3D12_FILTER ToDX12Filter(RHIFilter min_filter, RHIFilter mag_filter, RHIMipmapMode mip_mode, bool anisotropic)
{
	if (anisotropic)
	{
		return D3D12_FILTER_ANISOTROPIC;
	}
	else
	{
		if (min_filter == RHIFilter::Nearest && 
			mag_filter == RHIFilter::Nearest && 
			mip_mode == RHIMipmapMode::Nearest)
		{
			return D3D12_FILTER_MIN_MAG_MIP_POINT;
		}
		else if (min_filter == RHIFilter::Nearest &&
		         mag_filter == RHIFilter::Nearest &&
		         mip_mode == RHIMipmapMode::Linear)
		{
			return D3D12_FILTER_MIN_MAG_POINT_MIP_LINEAR;
		}
		else if (min_filter == RHIFilter::Nearest &&
		         mag_filter == RHIFilter::Linear &&
		         mip_mode == RHIMipmapMode::Nearest)
		{
			return D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT;
		}
		else if (min_filter == RHIFilter::Nearest &&
		         mag_filter == RHIFilter::Linear &&
		         mip_mode == RHIMipmapMode::Linear)
		{
			return D3D12_FILTER_MIN_POINT_MAG_MIP_LINEAR;
		}
		else if (min_filter == RHIFilter::Linear &&
		         mag_filter == RHIFilter::Nearest &&
		         mip_mode == RHIMipmapMode::Nearest)
		{
			return D3D12_FILTER_MIN_LINEAR_MAG_MIP_POINT;
		}
		else if (min_filter == RHIFilter::Linear &&
		         mag_filter == RHIFilter::Nearest &&
		         mip_mode == RHIMipmapMode::Linear)
		{
			return D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR;
		}
		else if (min_filter == RHIFilter::Linear &&
		         mag_filter == RHIFilter::Linear &&
		         mip_mode == RHIMipmapMode::Nearest)
		{
			return D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
		}
		else if (min_filter == RHIFilter::Linear &&
		         mag_filter == RHIFilter::Linear &&
		         mip_mode == RHIMipmapMode::Linear)
		{
			return D3D12_FILTER_MIN_MAG_MIP_LINEAR;
		}
	}
	return D3D12_FILTER_MIN_MAG_MIP_POINT;
}
}        // namespace Ilum::DX12