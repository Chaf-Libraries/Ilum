#pragma once

#include "RHIAccelerationStructure.hpp"
#include "RHIBuffer.hpp"
#include "RHICommand.hpp"
#include "RHIDescriptor.hpp"
#include "RHIDevice.hpp"
#include "RHIFrame.hpp"
#include "RHIPipelineState.hpp"
#include "RHIProfiler.hpp"
#include "RHIQueue.hpp"
#include "RHISampler.hpp"
#include "RHIShader.hpp"
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

	bool IsFeatureSupport(RHIFeature feature) const;

	void WaitIdle() const;

	RHISwapchain *GetSwapchain() const;

	std::unique_ptr<RHISwapchain> CreateSwapchain(void *window_handle, uint32_t width, uint32_t height, bool sync);

	// Create Texture
	std::unique_ptr<RHITexture> CreateTexture(const TextureDesc &desc, bool cuda = false);
	std::unique_ptr<RHITexture> CreateTexture2D(uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples = 1, bool cuda = false);
	std::unique_ptr<RHITexture> CreateTexture3D(uint32_t width, uint32_t height, uint32_t depth, RHIFormat format, RHITextureUsage usage, bool cuda = false);
	std::unique_ptr<RHITexture> CreateTextureCube(uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap, bool cuda = false);
	std::unique_ptr<RHITexture> CreateTexture2DArray(uint32_t width, uint32_t height, uint32_t layers, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples = 1, bool cuda = false);

	// Create Buffer
	std::unique_ptr<RHIBuffer> CreateBuffer(const BufferDesc &desc, bool cuda = false);
	std::unique_ptr<RHIBuffer> CreateBuffer(size_t size, RHIBufferUsage usage, RHIMemoryUsage memory, bool cuda = false);

	template <typename T>
	std::unique_ptr<RHIBuffer> CreateBuffer(size_t count, RHIBufferUsage usage, RHIMemoryUsage memory, bool cuda = false)
	{
		BufferDesc desc = {};
		desc.count      = count;
		desc.stride     = sizeof(T);
		desc.usage      = usage;
		desc.memory     = memory;
		desc.size       = desc.count * desc.stride;

		return CreateBuffer(desc, cuda);
	}

	// Create Sampler
	std::unique_ptr<RHISampler> CreateSampler(const SamplerDesc &desc, bool cuda = false);

	// Create Command
	RHICommand *CreateCommand(RHIQueueFamily family, bool cuda = false);

	// Create Descriptor
	std::unique_ptr<RHIDescriptor> CreateDescriptor(const ShaderMeta &meta, bool cuda = false);

	// Create PipelineState
	std::unique_ptr<RHIPipelineState> CreatePipelineState(bool cuda = false);

	// Create Shader
	std::unique_ptr<RHIShader> CreateShader(const std::string &entry_point, const std::vector<uint8_t> &source, bool cuda = false);

	// Create Render Target
	std::unique_ptr<RHIRenderTarget> CreateRenderTarget(bool cuda = false);

	// Create Profiler
	std::unique_ptr<RHIProfiler> CreateProfiler(bool cuda = false);

	// Create Fence
	std::unique_ptr<RHIFence> CreateFence(bool cuda = false);

	// Create Semaphore
	std::unique_ptr<RHISemaphore> CreateSemaphore(bool cuda = false);

	// Create Acceleration Structure
	std::unique_ptr<RHIAccelerationStructure> CreateAcccelerationStructure(bool cuda = false);

	// Get Queue
	RHIQueue *GetQueue(RHIQueueFamily family);

	RHIQueue *GetCUDAQueue();

	std::unique_ptr<RHIQueue> CreateQueue(RHIQueueFamily family, uint32_t idx = 0, bool cuda = false);

	// Get Back Buffer
	RHITexture *GetBackBuffer();

	// Resource conversion between different rhi backend
	RHITexture *ResourceConvert(RHITexture *src_texture, RHIBackend target);
	RHIBuffer  *ResourceConvert(RHIBuffer *src_buffer, RHIBackend target);

	// Frame
	void BeginFrame();
	void EndFrame();

  private:
	uint32_t m_current_frame = 0;

  private:
	Window *p_window = nullptr;

	std::unique_ptr<RHIDevice> m_device      = nullptr;
	std::unique_ptr<RHIDevice> m_cuda_device = nullptr;

	std::unique_ptr<RHISwapchain> m_swapchain = nullptr;

	std::unique_ptr<RHIQueue> m_graphics_queue = nullptr;
	std::unique_ptr<RHIQueue> m_compute_queue  = nullptr;
	std::unique_ptr<RHIQueue> m_transfer_queue = nullptr;
	std::unique_ptr<RHIQueue> m_cuda_queue     = nullptr;

	std::vector<std::unique_ptr<RHISemaphore>> m_present_complete;
	std::vector<std::unique_ptr<RHISemaphore>> m_render_complete;

	std::vector<std::unique_ptr<RHIFrame>> m_frames;
	std::vector<std::unique_ptr<RHIFrame>> m_cuda_frames;
};
}        // namespace Ilum