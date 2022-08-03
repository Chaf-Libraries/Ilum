#include "Texture.hpp"
#include "Definitions.hpp"
#include "Device.hpp"

namespace Ilum::DX12
{
inline D3D12_RESOURCE_DIMENSION GetResourceDimension(const TextureDesc &desc)
{
	if (desc.width > 1)
	{
		if (desc.height > 1)
		{
			if (desc.depth > 1)
			{
				return D3D12_RESOURCE_DIMENSION_TEXTURE3D;
			}
			else
			{
				return D3D12_RESOURCE_DIMENSION_TEXTURE2D;
			}
		}
		else
		{
			return D3D12_RESOURCE_DIMENSION_TEXTURE1D;
		}
	}
	return D3D12_RESOURCE_DIMENSION_UNKNOWN;
}

TextureState TextureState::Create(RHITextureState state)
{
	TextureState dx_state = {};

	switch (state)
	{
		case RHITextureState::Undefined:
			dx_state.state = D3D12_RESOURCE_STATE_COMMON;
			break;
		case RHITextureState::TransferSource:
			dx_state.state = D3D12_RESOURCE_STATE_COPY_SOURCE;
			break;
		case RHITextureState::TransferDest:
			dx_state.state = D3D12_RESOURCE_STATE_COPY_DEST;
			break;
		case RHITextureState::ShaderResource:
			dx_state.state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
			break;
		case RHITextureState::UnorderAccess:
			dx_state.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
			break;
		case RHITextureState::RenderTarget:
			dx_state.state = D3D12_RESOURCE_STATE_RENDER_TARGET;
			break;
		case RHITextureState::DepthWrite:
			dx_state.state = D3D12_RESOURCE_STATE_DEPTH_WRITE;
			break;
		case RHITextureState::DepthRead:
			dx_state.state = D3D12_RESOURCE_STATE_DEPTH_READ;
			break;
		case RHITextureState::Present:
			dx_state.state = D3D12_RESOURCE_STATE_PRESENT;
			break;
		default:
			dx_state.state = D3D12_RESOURCE_STATE_COMMON;
			break;
	}

	return dx_state;
}

Texture::Texture(RHIDevice *device, const TextureDesc &desc) :
    RHITexture(device, desc)
{
	D3D12_RESOURCE_DESC d3d12_desc = {};
	d3d12_desc.Dimension           = GetResourceDimension(desc);
	d3d12_desc.Alignment           = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
	d3d12_desc.Width               = desc.width;
	d3d12_desc.Height              = desc.height;
	d3d12_desc.DepthOrArraySize    = d3d12_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? desc.depth : desc.layers;
	d3d12_desc.MipLevels           = desc.mips;
	d3d12_desc.Format              = ToDX12Format[desc.format];
	d3d12_desc.SampleDesc.Count    = desc.samples;
	d3d12_desc.SampleDesc.Quality  = 0;
	d3d12_desc.Layout              = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	d3d12_desc.Flags               = ToDX12ResourceFlags(desc.usage);

	if (desc.usage & RHITextureUsage::RenderTarget)
	{
		d3d12_desc.Flags |= IsDepthFormat(desc.format) ?
		                        D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL :
                                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
	}

	D3D12MA::ALLOCATION_DESC alloc_desc = {};
	alloc_desc.HeapType                 = D3D12_HEAP_TYPE_DEFAULT;

	static_cast<Device *>(p_device)->GetAllocator()->CreateResource(
	    &alloc_desc, &d3d12_desc,
	    D3D12_RESOURCE_STATE_COMMON, NULL,
	    &m_allocation, IID_PPV_ARGS(&m_handle));
}

Texture::Texture(RHIDevice *device, const TextureDesc &desc, ComPtr<ID3D12Resource> &&texture) :
    RHITexture(device, desc), m_handle(texture)
{
}

Texture::~Texture()
{
	if (m_allocation)
	{
		m_allocation->Release();
		m_allocation = nullptr;
	}
}

ComPtr<ID3D12Resource> &Texture::GetHandle()
{
	return m_handle;
}
}        // namespace Ilum::DX12