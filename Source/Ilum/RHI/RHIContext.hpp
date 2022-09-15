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

	void WaitIdle() const;

	RHISwapchain *GetSwapchain() const;

	std::unique_ptr<RHISwapchain> CreateSwapchain(void *window_handle, uint32_t width, uint32_t height, bool sync);

	// Create Texture
	std::unique_ptr<RHITexture> CreateTexture(const TextureDesc &desc, RHIBackend backend = RHIBackend::Vulkan);
	std::unique_ptr<RHITexture> CreateTexture2D(uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples = 1, RHIBackend backend = RHIBackend::Vulkan);
	std::unique_ptr<RHITexture> CreateTexture3D(uint32_t width, uint32_t height, uint32_t depth, RHIFormat format, RHITextureUsage usage, RHIBackend backend = RHIBackend::Vulkan);
	std::unique_ptr<RHITexture> CreateTextureCube(uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap, RHIBackend backend = RHIBackend::Vulkan);
	std::unique_ptr<RHITexture> CreateTexture2DArray(uint32_t width, uint32_t height, uint32_t layers, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples = 1, RHIBackend backend = RHIBackend::Vulkan);

	// Create Buffer
	std::unique_ptr<RHIBuffer> CreateBuffer(const BufferDesc &desc, RHIBackend backend = RHIBackend::Vulkan);
	std::unique_ptr<RHIBuffer> CreateBuffer(size_t size, RHIBufferUsage usage, RHIMemoryUsage memory, RHIBackend backend = RHIBackend::Vulkan);

	template <typename T>
	std::unique_ptr<RHIBuffer> CreateBuffer(size_t count, RHIBufferUsage usage, RHIMemoryUsage memory, RHIBackend backend = RHIBackend::Vulkan)
	{
		BufferDesc desc = {};
		desc.count      = count;
		desc.stride     = sizeof(T);
		desc.usage      = usage;
		desc.memory     = memory;
		desc.size       = desc.count * desc.stride;

		return CreateBuffer(desc, backend);
	}

	// Create Sampler
	std::unique_ptr<RHISampler> CreateSampler(const SamplerDesc &desc, RHIBackend backend = RHIBackend::Vulkan);

	// Create Command
	RHICommand *CreateCommand(RHIQueueFamily family, RHIBackend backend = RHIBackend::Vulkan);

	// Create Descriptor
	std::unique_ptr<RHIDescriptor> CreateDescriptor(const ShaderMeta &meta, RHIBackend backend = RHIBackend::Vulkan);

	// Create PipelineState
	std::unique_ptr<RHIPipelineState> CreatePipelineState(RHIBackend backend = RHIBackend::Vulkan);

	// Create Shader
	std::unique_ptr<RHIShader> CreateShader(const std::string &entry_point, const std::vector<uint8_t> &source, RHIBackend backend = RHIBackend::Vulkan);

	// Create Render Target
	std::unique_ptr<RHIRenderTarget> CreateRenderTarget(RHIBackend backend = RHIBackend::Vulkan);

	// Create Profiler
	std::unique_ptr<RHIProfiler> CreateProfiler(RHIBackend backend = RHIBackend::Vulkan);

	// Create Fence
	std::unique_ptr<RHIFence> CreateFence(RHIBackend backend = RHIBackend::Vulkan);

	// Create Semaphore
	std::unique_ptr<RHISemaphore> CreateSemaphore(RHIBackend backend = RHIBackend::Vulkan);

	// Create Acceleration Structure
	std::unique_ptr<RHIAccelerationStructure> CreateAcccelerationStructure(RHIBackend backend = RHIBackend::Vulkan);

	// Get Queue
	RHIQueue *GetQueue(RHIQueueFamily family);

	RHIQueue *GetCUDAQueue();

	std::unique_ptr<RHIQueue> CreateQueue(RHIQueueFamily family, uint32_t idx = 0, RHIBackend backend = RHIBackend::Vulkan);

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

	std::map<RHIBackend, RHIDevice *> m_devices;

	std::unique_ptr<RHIDevice> m_vulkan_device = nullptr;
	std::unique_ptr<RHIDevice> m_dx12_device   = nullptr;
	std::unique_ptr<RHIDevice> m_cuda_device   = nullptr;

	RHIDevice *m_device = nullptr;

	std::unique_ptr<RHISwapchain> m_swapchain = nullptr;

	std::unique_ptr<RHIQueue> m_graphics_queue = nullptr;
	std::unique_ptr<RHIQueue> m_compute_queue  = nullptr;
	std::unique_ptr<RHIQueue> m_transfer_queue = nullptr;
	std::unique_ptr<RHIQueue> m_cuda_queue     = nullptr;

	std::vector<std::unique_ptr<RHISemaphore>> m_present_complete;
	std::vector<std::unique_ptr<RHISemaphore>> m_render_complete;

	std::vector<std::unique_ptr<RHIFrame>> m_frames;
};
}        // namespace Ilum