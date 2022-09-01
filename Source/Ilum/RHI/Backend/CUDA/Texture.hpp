#pragma once

#include "RHI/RHITexture.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

namespace Ilum::CUDA
{
class Texture : public RHITexture
{
  public:
	Texture(RHIDevice *device, const TextureDesc &desc);

	Texture(RHIDevice *device, cudaArray_t cuda_array, const TextureDesc &desc);

	virtual ~Texture() override;

	uint64_t GetSurfaceHandle() const;

	uint64_t GetTextureHandle() const;

  private:
	cudaSurfaceObject_t m_surface_handle = 0;
	cudaTextureObject_t m_texture_handle = 0;

	cudaArray_t m_array = {};

	bool m_is_backbuffer = false;
};
}        // namespace Ilum::CUDA