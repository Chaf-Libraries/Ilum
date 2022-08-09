#pragma once

#include "RHIBuffer.hpp"
#include "RHICommand.hpp"
#include "RHIDevice.hpp"
#include "RHIFrame.hpp"
#include "RHIQueue.hpp"
#include "RHISampler.hpp"
#include "RHISwapchain.hpp"
#include "RHISynchronization.hpp"
#include "RHITexture.hpp"

namespace Ilum
{
class RHIContext
{
  public:
	RHIContext(Window *window);
	~RHIContext();

	RHIBackend GetBackend() const;

	// Create Texture
	std::unique_ptr<RHITexture> CreateTexture(const TextureDesc &desc);
	std::unique_ptr<RHITexture> CreateTexture2D(uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples = 1);
	std::unique_ptr<RHITexture> CreateTexture3D(uint32_t width, uint32_t height, uint32_t depth, RHIFormat format, RHITextureUsage usage);
	std::unique_ptr<RHITexture> CreateTextureCube(uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap);
	std::unique_ptr<RHITexture> CreateTexture2DArray(uint32_t width, uint32_t height, uint32_t layers, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples = 1);

	// Create Buffer
	std::unique_ptr<RHIBuffer> CreateBuffer(const BufferDesc &desc);

	// Create Sampler
	std::unique_ptr<RHISampler> CreateSampler(const SamplerDesc &desc);

	// Create Command
	RHICommand *CreateCommand(RHIQueueFamily family);

	// Get Queue
	RHIQueue *GetQueue(RHIQueueFamily family);

	RHITexture *GetBackBuffer();

	// Frame
	void BeginFrame();
	void EndFrame();

  private:
	uint32_t m_current_frame = 0;

  private:
	Window *p_window = nullptr;

	std::unique_ptr<RHIDevice>    m_device    = nullptr;
	std::unique_ptr<RHISwapchain> m_swapchain = nullptr;

	std::unique_ptr<RHIQueue> m_graphics_queue = nullptr;
	std::unique_ptr<RHIQueue> m_compute_queue  = nullptr;
	std::unique_ptr<RHIQueue> m_transfer_queue = nullptr;

	std::vector<std::unique_ptr<RHISemaphore>> m_present_complete;
	std::vector<std::unique_ptr<RHISemaphore>> m_render_complete;

	std::vector<std::unique_ptr<RHIFrame>> m_frames;
};
}        // namespace Ilum