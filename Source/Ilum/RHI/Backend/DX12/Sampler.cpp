#include "Sampler.hpp"
#include "Definitions.hpp"

namespace Ilum::DX12
{
Sampler::Sampler(RHIDevice *device, const SamplerDesc &desc):
    RHISampler(device, desc)
{
	m_handle.Filter           = ToDX12Filter(desc.min_filter, desc.mag_filter, desc.mipmap_mode, desc.anisotropic);
	m_handle.AddressU         = ToDX12AddressMode[desc.address_mode_u];
	m_handle.AddressV         = ToDX12AddressMode[desc.address_mode_v];
	m_handle.AddressW         = ToDX12AddressMode[desc.address_mode_w];
	m_handle.MipLODBias       = 0;
	m_handle.MaxAnisotropy    = 1000;
	m_handle.ComparisonFunc   = D3D12_COMPARISON_FUNC_NEVER;
	m_handle.BorderColor      = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
	m_handle.MinLOD           = desc.min_lod;
	m_handle.MaxLOD           = desc.max_lod;
	m_handle.ShaderRegister   = 0;
	m_handle.RegisterSpace    = 0;
	m_handle.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
}

}        // namespace Ilum::DX12