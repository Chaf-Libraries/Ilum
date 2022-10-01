#include "Texture.hpp"
#include "Backend/Vulkan/Texture.hpp"

namespace Ilum::CUDA
{
cudaChannelFormatDesc GetCUDAChannelFormatDesc(RHIFormat format)
{
	cudaChannelFormatDesc desc = {};

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

HANDLE GetVkImageMemHandle(Vulkan::Device *device, Vulkan::Texture *texture, VkExternalMemoryHandleTypeFlagBitsKHR external_memory_handle_type_flag)
{
	HANDLE handle = {};

	if (!texture->GetDesc().external)
	{
		LOG_ERROR("Texture {} is not external accessable!", texture->GetDesc().name);
		return handle;
	}

	VkMemoryGetWin32HandleInfoKHR handle_info = {};

	handle_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
	handle_info.pNext      = NULL;
	handle_info.memory     = texture->GetMemory();
	handle_info.handleType = external_memory_handle_type_flag;

	device->GetMemoryWin32Handle(&handle_info, &handle);

	return handle;
}

Texture::Texture(Device *device, Vulkan::Device *vk_device, Vulkan::Texture *vk_texture) :
    RHITexture(device, vk_texture->GetDesc())
{
	cudaExternalMemoryHandleDesc cuda_external_memory_handle_desc = {};

#ifdef _WIN64
	cuda_external_memory_handle_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
	cuda_external_memory_handle_desc.handle.win32.handle = GetVkImageMemHandle(vk_device, vk_texture, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT);
#else
	cuda_external_memory_handle_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
	cuda_external_memory_handle_desc.handle.fd = GetVkImageMemHandle(device, texture, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);
#endif
	cuda_external_memory_handle_desc.size = vk_texture->GetMemorySize();

	cudaImportExternalMemory(&m_external_memory, &cuda_external_memory_handle_desc);

	cudaExtent            extent = {vk_texture->GetDesc().width, vk_texture->GetDesc().height, vk_texture->GetDesc().depth - 1};
	cudaChannelFormatDesc format = GetCUDAChannelFormatDesc(vk_texture->GetDesc().format);

	cudaExternalMemoryMipmappedArrayDesc cuda_external_memory_mipmapped_array_desc = {};

	cuda_external_memory_mipmapped_array_desc.offset     = 0;
	cuda_external_memory_mipmapped_array_desc.formatDesc = format;
	cuda_external_memory_mipmapped_array_desc.extent     = extent;
	cuda_external_memory_mipmapped_array_desc.flags      = 0;
	cuda_external_memory_mipmapped_array_desc.numLevels  = vk_texture->GetDesc().mips;

	cudaExternalMemoryGetMappedMipmappedArray(&m_mipmapped_array, m_external_memory, &cuda_external_memory_mipmapped_array_desc);

	for (uint32_t mip_level = 0; mip_level < vk_texture->GetDesc().mips; mip_level++)
	{
		cudaArray_t cuda_mip_level_array = {}, cuda_mip_level_array_orig = {};

		cudaResourceDesc resource_desc = {};

		cudaGetMipmappedArrayLevel(&cuda_mip_level_array, m_mipmapped_array, mip_level);

		uint32_t width  = (vk_texture->GetDesc().width >> mip_level) ? (vk_texture->GetDesc().width >> mip_level) : 1;
		uint32_t height = (vk_texture->GetDesc().height >> mip_level) ? (vk_texture->GetDesc().height >> mip_level) : 1;
		cudaMemcpy2DArrayToArray(cuda_mip_level_array_orig, 0, 0, cuda_mip_level_array, 0, 0, width * static_cast<size_t>(format.x + format.y + format.z + format.w) / 8ull, height, cudaMemcpyDeviceToDevice);

		resource_desc.resType         = cudaResourceTypeArray;
		resource_desc.res.array.array = cuda_mip_level_array;

		cudaSurfaceObject_t surface = {};
		cudaCreateSurfaceObject(&surface, &resource_desc);

		m_surfaces.push_back(surface);
	}

	cudaResourceDesc cuda_resource_desc = {};

	cuda_resource_desc.resType           = cudaResourceTypeMipmappedArray;
	cuda_resource_desc.res.mipmap.mipmap = m_mipmapped_array;

	cudaTextureDesc cuda_texture_desc = {};

	cuda_texture_desc.normalizedCoords    = true;
	cuda_texture_desc.filterMode          = cudaFilterModeLinear;
	cuda_texture_desc.mipmapFilterMode    = cudaFilterModeLinear;
	cuda_texture_desc.addressMode[0]      = cudaAddressModeClamp;
	cuda_texture_desc.addressMode[1]      = cudaAddressModeClamp;
	cuda_texture_desc.maxMipmapLevelClamp = static_cast<float>(m_desc.mips) - 1.f;
	cuda_texture_desc.readMode            =  cudaReadModeElementType;

	cudaCreateTextureObject(&m_texture_handle, &cuda_resource_desc, &cuda_texture_desc, nullptr);

	cudaMalloc((void **) &m_surface_list, sizeof(cudaSurfaceObject_t) * m_desc.mips);
	cudaMemcpy(m_surface_list, m_surfaces.data(), sizeof(cudaSurfaceObject_t) * m_desc.mips, cudaMemcpyHostToDevice);
}

Texture::~Texture()
{
	cudaDestroyTextureObject(m_texture_handle);

	for (auto &surface : m_surfaces)
	{
		cudaDestroySurfaceObject(surface);
	}
	m_surfaces.clear();

	if (m_surface_list)
	{
		cudaFree(m_surface_list);
	}

	if (m_external_memory)
	{
		cudaFreeMipmappedArray(m_mipmapped_array);
		cudaDestroyExternalMemory(m_external_memory);
	}
}

const cudaSurfaceObject_t *Texture::GetSurfaceDeviceHandle() const
{
	return m_surface_list;
}

const std::vector<cudaSurfaceObject_t> &Texture::GetSurfaceHostHandle() const
{
	return m_surfaces;
}

const cudaTextureObject_t *Texture::GetTextureHandle() const
{
	return &m_texture_handle;
}
}        // namespace Ilum::CUDA