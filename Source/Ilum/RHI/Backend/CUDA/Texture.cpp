#include "Texture.hpp"

namespace Ilum::CUDA
{
cudaChannelFormatDesc GetCUDAChannelFormatDesc(RHIFormat format)
{
	cudaChannelFormatDesc desc = {};
	std::memset(&desc, 0, sizeof(desc));

	switch (format)
	{
		case RHIFormat::Undefined:
			desc = cudaChannelFormatDesc{0, 0, 0, 0, cudaChannelFormatKindNone};
			break;
		case RHIFormat::R16_UINT:
			desc = cudaChannelFormatDesc{16, 0, 0, 0, cudaChannelFormatKindUnsigned};
			break;
		case RHIFormat::R16_SINT:
			desc = cudaChannelFormatDesc{16, 0, 0, 0, cudaChannelFormatKindSigned};
			break;
		case RHIFormat::R16_FLOAT:
			desc = cudaChannelFormatDesc{16, 0, 0, 0, cudaChannelFormatKindFloat};
			break;
		case RHIFormat::R8G8B8A8_UNORM:
			desc = cudaChannelFormatDesc{8, 8, 8, 8, cudaChannelFormatKindUnsigned};
			break;
		case RHIFormat::B8G8R8A8_UNORM:
			desc = cudaChannelFormatDesc{8, 8, 8, 8, cudaChannelFormatKindUnsigned};
			break;
		case RHIFormat::R32_UINT:
			desc = cudaChannelFormatDesc{32, 0, 0, 0, cudaChannelFormatKindUnsigned};
			break;
		case RHIFormat::R32_SINT:
			desc = cudaChannelFormatDesc{32, 0, 0, 0, cudaChannelFormatKindSigned};
			break;
		case RHIFormat::R32_FLOAT:
			desc = cudaChannelFormatDesc{32, 0, 0, 0, cudaChannelFormatKindFloat};
			break;
		case RHIFormat::D32_FLOAT:
			desc = cudaChannelFormatDesc{32, 0, 0, 0, cudaChannelFormatKindFloat};
			break;
		case RHIFormat::R16G16_UINT:
			desc = cudaChannelFormatDesc{16, 16, 0, 0, cudaChannelFormatKindUnsigned};
			break;
		case RHIFormat::R16G16_SINT:
			desc = cudaChannelFormatDesc{16, 16, 0, 0, cudaChannelFormatKindSigned};
			break;
		case RHIFormat::R16G16_FLOAT:
			desc = cudaChannelFormatDesc{16, 16, 0, 0, cudaChannelFormatKindFloat};
			break;
		case RHIFormat::R16G16B16A16_UINT:
			desc = cudaChannelFormatDesc{16, 16, 16, 16, cudaChannelFormatKindUnsigned};
			break;
		case RHIFormat::R16G16B16A16_SINT:
			desc = cudaChannelFormatDesc{16, 16, 16, 16, cudaChannelFormatKindSigned};
			break;
		case RHIFormat::R16G16B16A16_FLOAT:
			desc = cudaChannelFormatDesc{16, 16, 16, 16, cudaChannelFormatKindFloat};
			break;
		case RHIFormat::R32G32_UINT:
			desc = cudaChannelFormatDesc{32, 32, 0, 0, cudaChannelFormatKindUnsigned};
			break;
		case RHIFormat::R32G32_SINT:
			desc = cudaChannelFormatDesc{32, 32, 0, 0, cudaChannelFormatKindSigned};
			break;
		case RHIFormat::R32G32_FLOAT:
			desc = cudaChannelFormatDesc{32, 32, 0, 0, cudaChannelFormatKindFloat};
			break;
		case RHIFormat::R32G32B32_UINT:
			desc = cudaChannelFormatDesc{32, 32, 32, 0, cudaChannelFormatKindUnsigned};
			break;
		case RHIFormat::R32G32B32_SINT:
			desc = cudaChannelFormatDesc{32, 32, 32, 0, cudaChannelFormatKindSigned};
			break;
		case RHIFormat::R32G32B32_FLOAT:
			desc = cudaChannelFormatDesc{32, 32, 32, 0, cudaChannelFormatKindFloat};
			break;
		case RHIFormat::R32G32B32A32_UINT:
			desc = cudaChannelFormatDesc{32, 32, 32, 32, cudaChannelFormatKindUnsigned};
			break;
		case RHIFormat::R32G32B32A32_SINT:
			desc = cudaChannelFormatDesc{32, 32, 32, 32, cudaChannelFormatKindSigned};
			break;
		case RHIFormat::R32G32B32A32_FLOAT:
			desc = cudaChannelFormatDesc{32, 32, 32, 32, cudaChannelFormatKindFloat};
			break;
		default:
			break;
	}

	return desc;
}

Texture::Texture(RHIDevice *device, const TextureDesc &desc) :
    RHITexture(device, desc)
{
	cudaChannelFormatDesc channel_format_desc = GetCUDAChannelFormatDesc(desc.format);
	cudaMallocArray(&m_array, &channel_format_desc, desc.width, desc.height);

	cudaResourceDesc res_desc = {};
	std::memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType         = cudaResourceTypeArray;
	res_desc.res.array.array = m_array;

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.addressMode[0]   = cudaAddressModeWrap;
	tex_desc.addressMode[1]   = cudaAddressModeWrap;
	tex_desc.filterMode       = cudaFilterModeLinear;
	tex_desc.readMode         = cudaReadModeElementType;
	tex_desc.normalizedCoords = 1;

	cudaCreateTextureObject(&m_texture_handle, &res_desc, &tex_desc, NULL);

	cudaCreateSurfaceObject(&m_surface_handle, &res_desc);
}

Texture::Texture(RHIDevice *device, cudaArray_t cuda_array, const TextureDesc &desc) :
    RHITexture(device, desc), m_array(cuda_array), m_is_backbuffer(true)
{
	cudaResourceDesc res_desc = {};
	std::memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType         = cudaResourceTypeArray;
	res_desc.res.array.array = m_array;

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.addressMode[0]   = cudaAddressModeWrap;
	tex_desc.addressMode[1]   = cudaAddressModeWrap;
	tex_desc.filterMode       = cudaFilterModeLinear;
	tex_desc.readMode         = cudaReadModeElementType;
	tex_desc.normalizedCoords = 1;

	cudaCreateTextureObject(&m_texture_handle, &res_desc, &tex_desc, NULL);

	cudaCreateSurfaceObject(&m_surface_handle, &res_desc);
}

Texture::~Texture()
{
	cudaDestroySurfaceObject(m_surface_handle);
	cudaDestroyTextureObject(m_texture_handle);

	if (!m_is_backbuffer)
	{
		cudaFreeArray(m_array);
	}
}

uint64_t Texture::GetSurfaceHandle() const
{
	return m_surface_handle;
}

uint64_t Texture::GetTextureHandle() const
{
	return m_texture_handle;
}
}        // namespace Ilum::CUDA