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

	virtual ~Texture() override;

	uint64_t GetHandle() const;

  private:
	uint64_t m_handle = 0;

	cudaSurfaceObject_t m_surface_handle = 0;
	cudaTextureObject_t m_texture_handle = 0;

	cudaArray_t m_array = {};
};
}        // namespace Ilum::CUDA