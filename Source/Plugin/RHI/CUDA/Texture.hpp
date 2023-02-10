#pragma once

#include "Fwd.hpp"

namespace Ilum::CUDA
{
class Texture : public RHITexture
{
  public:
	Texture(RHIDevice *device, const TextureDesc &desc, HANDLE mem_handle, size_t memory_size);

	virtual ~Texture() override;

	virtual size_t GetMemorySize() const override;

	const cudaSurfaceObject_t *GetSurfaceDeviceHandle() const;

	const std::vector<cudaSurfaceObject_t> &GetSurfaceHostHandle() const;

	const cudaTextureObject_t *GetTextureHandle() const;

  private:
	cudaTextureObject_t m_texture_handle;

	std::vector<cudaSurfaceObject_t> m_surfaces;
	cudaSurfaceObject_t             *m_surface_list = nullptr;

	cudaExternalMemory_t m_external_memory      = nullptr;
	cudaMipmappedArray_t m_mipmapped_array      = nullptr;

	size_t m_memory_size = 0;

	bool m_is_backbuffer = false;
};
}        // namespace Ilum::CUDA