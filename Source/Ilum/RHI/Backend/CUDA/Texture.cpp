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

}

Texture::~Texture()
{
}

uint64_t Texture::GetHandle() const
{
	return m_handle;
}
}        // namespace Ilum::CUDA