#pragma once

#include "RHI/RHITexture.hpp"
#include "Device.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

namespace Ilum::Vulkan
{
class Texture;
class Device;
}        // namespace Ilum::Vulkan

namespace Ilum::CUDA
{
class Texture : public RHITexture
{
  public:
	Texture(Device *device, const TextureDesc &desc);

	Texture(Device *device, cudaArray_t cuda_array, const TextureDesc &desc);

	Texture(Device *device, Vulkan::Device *vk_device, Vulkan::Texture *vk_texture);

	virtual ~Texture() override;

	uint64_t GetSurfaceHandle() const;

	uint64_t GetTextureHandle() const;

  private:
	cudaSurfaceObject_t m_surface_handle = 0;
	cudaTextureObject_t m_texture_handle = 0;

	std::vector<cudaSurfaceObject_t> m_surfaces;
	cudaSurfaceObject_t *m_surface_list = nullptr;

	cudaExternalMemory_t m_external_memory        = nullptr;
	cudaMipmappedArray_t m_mipmapped_array        = nullptr;
	cudaMipmappedArray_t m_mipmapped_array_orig        = nullptr;



	cudaArray_t m_array = {};

	bool m_is_backbuffer = false;
};
}        // namespace Ilum::CUDA