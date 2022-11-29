#pragma once

#include "Fwd.hpp"

#include "RHIAccelerationStructure.hpp"
#include "RHIBuffer.hpp"
#include "RHICommand.hpp"
#include "RHIDescriptor.hpp"
#include "RHIDevice.hpp"
#include "RHIFrame.hpp"
#include "RHIPipelineState.hpp"
#include "RHIProfiler.hpp"
#include "RHIQueue.hpp"
#include "RHIRenderTarget.hpp"
#include "RHISampler.hpp"
#include "RHIShader.hpp"
#include "RHISwapchain.hpp"
#include "RHISynchronization.hpp"
#include "RHITexture.hpp"

namespace Ilum
{
#undef CreateSemaphore

class RHIContext
{
  public:
	RHIContext(Window *window, const std::string &backend = "Vulkan", bool vsync = true);

	~RHIContext();

	const std::string &GetDeviceName() const;

	const std::string GetBackend() const;

	bool HasCUDA() const;

	bool IsVsync() const;

	void SetVsync(bool vsync);

	bool IsFeatureSupport(RHIFeature feature) const;

	void WaitIdle() const;

	RHISwapchain *GetSwapchain() const;

	std::unique_ptr<RHISwapchain> CreateSwapchain(void *window_handle, uint32_t width, uint32_t height, bool sync);

	// Create Texture
	std::unique_ptr<RHITexture> CreateTexture(const TextureDesc &desc);
	std::unique_ptr<RHITexture> CreateTexture2D(uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples = 1);
	std::unique_ptr<RHITexture> CreateTexture3D(uint32_t width, uint32_t height, uint32_t depth, RHIFormat format, RHITextureUsage usage);
	std::unique_ptr<RHITexture> CreateTextureCube(uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap);
	std::unique_ptr<RHITexture> CreateTexture2DArray(uint32_t width, uint32_t height, uint32_t layers, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples = 1);

	// Texture Conversion
	std::unique_ptr<RHITexture> MapToCUDATexture(RHITexture *texture);

	// Create Buffer
	std::unique_ptr<RHIBuffer> CreateBuffer(const BufferDesc &desc);
	std::unique_ptr<RHIBuffer> CreateBuffer(size_t size, RHIBufferUsage usage, RHIMemoryUsage memory);

	template <typename T>
	std::unique_ptr<RHIBuffer> CreateBuffer(size_t count, RHIBufferUsage usage, RHIMemoryUsage memory)
	{
		BufferDesc desc = {};
		desc.count      = count;
		desc.stride     = sizeof(T);
		desc.usage      = usage;
		desc.memory     = memory;
		desc.size       = desc.count * desc.stride;

		return CreateBuffer(desc);
	}

	// Buffer Conversion
	std::unique_ptr<RHIBuffer> MapToCUDABuffer(RHIBuffer *buffer);

	// Create Sampler
	RHISampler *CreateSampler(const SamplerDesc &desc);

	// Create Command
	RHICommand *CreateCommand(RHIQueueFamily family, bool cuda = false);

	// Create Descriptor
	RHIDescriptor *CreateDescriptor(const ShaderMeta &meta, bool cuda = false);

	// Create PipelineState
	std::unique_ptr<RHIPipelineState> CreatePipelineState(bool cuda = false);

	// Create Shader
	std::unique_ptr<RHIShader> CreateShader(const std::string &entry_point, const std::vector<uint8_t> &source, bool cuda = false);

	// Create Render Target
	std::unique_ptr<RHIRenderTarget> CreateRenderTarget(bool cuda = false);

	// Create Profiler
	std::unique_ptr<RHIProfiler> CreateProfiler(bool cuda = false);

	// Create Fence
	std::unique_ptr<RHIFence> CreateFence();

	// Create Fence that will reset every frame
	RHIFence *CreateFrameFence();

	// Create Semaphore
	std::unique_ptr<RHISemaphore> CreateSemaphore(bool cuda = false);

	// Create Frame Semaphore
	RHISemaphore *CreateFrameSemaphore();

	std::unique_ptr<RHISemaphore> MapToCUDASemaphore(RHISemaphore *semaphore);

	// Create Acceleration Structure
	std::unique_ptr<RHIAccelerationStructure> CreateAcccelerationStructure();

	// Submit command buffer
	void Submit(std::vector<RHICommand *> &&cmd_buffers, std::vector<RHISemaphore *> &&wait_semaphores = {}, std::vector<RHISemaphore *> &&signal_semaphores = {});

	// Execute immediate command buffer
	void Execute(RHICommand *cmd_buffer);

	void Execute(std::vector<RHICommand *> &&cmd_buffers, std::vector<RHISemaphore *> &&wait_semaphores, std::vector<RHISemaphore *> &&signal_semaphores, RHIFence *fence = nullptr);

	void Reset();

	// Get Back Buffer
	RHITexture *GetBackBuffer();

	// Frame
	void BeginFrame();

	void EndFrame();

  private:
	uint32_t m_current_frame = 0;
	bool     m_vsync         = false;

  private:
	Window *p_window = nullptr;

	std::unique_ptr<RHIDevice> m_device      = nullptr;
	std::unique_ptr<RHIDevice> m_cuda_device = nullptr;

	std::unique_ptr<RHISwapchain> m_swapchain = nullptr;

	std::unique_ptr<RHIQueue> m_queue      = nullptr;
	std::unique_ptr<RHIQueue> m_cuda_queue = nullptr;

	std::vector<std::unique_ptr<RHISemaphore>> m_present_complete;
	std::vector<std::unique_ptr<RHISemaphore>> m_render_complete;

	std::vector<std::unique_ptr<RHIFrame>> m_frames;
	std::vector<std::unique_ptr<RHIFrame>> m_cuda_frames;

	std::vector<SubmitInfo> m_submit_infos;

	std::unordered_map<size_t, std::unique_ptr<RHISampler>> m_samplers;
};
}        // namespace Ilum