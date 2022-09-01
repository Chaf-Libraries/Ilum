#pragma once

#include "RHI/RHISwapchain.hpp"

#include "Texture.hpp"

#include <glad/glad.h>

#include <cuda_gl_interop.h>

namespace Ilum::CUDA
{
class Swapchain : public RHISwapchain
{
  public:
	Swapchain(RHIDevice *device, void *window_handle, uint32_t width, uint32_t height, bool vsync = false);

	virtual ~Swapchain() override;

	virtual uint32_t GetTextureCount() override;

	virtual void AcquireNextTexture(RHISemaphore *signal_semaphore, RHIFence *signal_fence) override;

	virtual RHITexture *GetCurrentTexture() override;

	virtual uint32_t GetCurrentFrameIndex() override;

	virtual bool Present(RHISemaphore *semaphore) override;

	virtual void Resize(uint32_t width, uint32_t height) override;

  private:
	void *p_window = nullptr;

	GLuint m_gl_framebuffers[2];
	GLuint m_gl_renderbuffers[2];

	cudaGraphicsResource *m_cuda_handles[2];

	std::array<std::unique_ptr<Texture>, 2> m_textures;

	cudaStream_t m_stream;

	GLuint                m_gl_handle   = 0;
	cudaGraphicsResource *m_cuda_handle = nullptr;

	uint32_t m_frame_index = 0;
};
}        // namespace Ilum::CUDA