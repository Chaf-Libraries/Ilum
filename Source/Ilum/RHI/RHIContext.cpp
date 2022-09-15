#include "RHIContext.hpp"
#include <Core/Time.hpp>

namespace Ilum
{
RHIContext::RHIContext(Window *window) :
    p_window(window)
{
	m_vulkan_device = RHIDevice::Create(RHIBackend::Vulkan);
	m_dx12_device   = RHIDevice::Create(RHIBackend::DX12);
	m_cuda_device   = RHIDevice::Create(RHIBackend::CUDA);

	m_devices = {
	    {RHIBackend::Vulkan, m_vulkan_device.get()},
	    {RHIBackend::DX12, m_dx12_device.get()},
	    {RHIBackend::CUDA, m_cuda_device.get()},
	};

#ifdef RHI_BACKEND_VULKAN
	m_device = m_vulkan_device.get();
	LOG_INFO("RHI Backend: Vulkan");
#elif RHI_BACKEND_DX12
	LOG_INFO("RHI Backend: DX12");
#endif        // RHI_BACKEND

	m_swapchain = RHISwapchain::Create(m_device, p_window->GetNativeHandle(), p_window->GetWidth(), p_window->GetHeight(), false);

	m_graphics_queue = RHIQueue::Create(m_device, RHIQueueFamily::Graphics);
	m_compute_queue  = RHIQueue::Create(m_device, RHIQueueFamily::Compute);
	m_transfer_queue = RHIQueue::Create(m_device, RHIQueueFamily::Transfer);
	m_cuda_queue     = RHIQueue::Create(m_cuda_device.get(), RHIQueueFamily::Compute);

	for (uint32_t i = 0; i < m_swapchain->GetTextureCount(); i++)
	{
		m_frames.emplace_back(RHIFrame::Create(m_device));
		m_present_complete.emplace_back(RHISemaphore::Create(m_device));
		m_render_complete.emplace_back(RHISemaphore::Create(m_device));
	}
}

RHIContext::~RHIContext()
{
	m_device->WaitIdle();

	for (auto &frame : m_frames)
	{
		frame->Reset();
	}
	m_frames.clear();

	m_present_complete.clear();
	m_render_complete.clear();

	m_swapchain.reset();
	m_vulkan_device.reset();
	m_dx12_device.reset();
	m_cuda_device.reset();
}

void RHIContext::WaitIdle() const
{
	m_device->WaitIdle();
}

RHISwapchain *RHIContext::GetSwapchain() const
{
	return m_swapchain.get();
}

std::unique_ptr<RHISwapchain> RHIContext::CreateSwapchain(void *window_handle, uint32_t width, uint32_t height, bool sync)
{
	return RHISwapchain::Create(m_device, window_handle, width, height, sync);
}

std::unique_ptr<RHITexture> RHIContext::CreateTexture(const TextureDesc &desc, RHIBackend backend)
{
	return RHITexture::Create(m_device, desc);
}

std::unique_ptr<RHITexture> RHIContext::CreateTexture2D(uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples, RHIBackend backend)
{
	return RHITexture::Create2D(m_device, width, height, format, usage, mipmap, samples);
}

std::unique_ptr<RHITexture> RHIContext::CreateTexture3D(uint32_t width, uint32_t height, uint32_t depth, RHIFormat format, RHITextureUsage usage, RHIBackend backend)
{
	return RHITexture::Create3D(m_device, width, height, depth, format, usage);
}

std::unique_ptr<RHITexture> RHIContext::CreateTextureCube(uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap, RHIBackend backend)
{
	return RHITexture::CreateCube(m_device, width, height, format, usage, mipmap);
}

std::unique_ptr<RHITexture> RHIContext::CreateTexture2DArray(uint32_t width, uint32_t height, uint32_t layers, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples, RHIBackend backend)
{
	return RHITexture::Create2DArray(m_device, width, height, layers, format, usage, mipmap, samples);
}

std::unique_ptr<RHIBuffer> RHIContext::CreateBuffer(const BufferDesc &desc, RHIBackend backend)
{
	return RHIBuffer::Create(m_device, desc);
}

std::unique_ptr<RHIBuffer> RHIContext::CreateBuffer(size_t size, RHIBufferUsage usage, RHIMemoryUsage memory, RHIBackend backend)
{
	BufferDesc desc = {};
	desc.size       = size;
	desc.usage      = usage;
	desc.memory     = memory;

	return RHIBuffer::Create(m_device, desc);
}

std::unique_ptr<RHISampler> RHIContext::CreateSampler(const SamplerDesc &desc, RHIBackend backend)
{
	return RHISampler::Create(m_device, desc);
}

RHICommand *RHIContext::CreateCommand(RHIQueueFamily family, RHIBackend backend)
{
	return m_frames[m_current_frame]->AllocateCommand(family);
}

std::unique_ptr<RHIDescriptor> RHIContext::CreateDescriptor(const ShaderMeta &meta, RHIBackend backend)
{
	return RHIDescriptor::Create(m_device, meta);
}

std::unique_ptr<RHIPipelineState> RHIContext::CreatePipelineState(RHIBackend backend)
{
	return RHIPipelineState::Create(m_device);
}

std::unique_ptr<RHIShader> RHIContext::CreateShader(const std::string &entry_point, const std::vector<uint8_t> &source, RHIBackend backend)
{
	return RHIShader::Create(m_device, entry_point, source);
}

std::unique_ptr<RHIRenderTarget> RHIContext::CreateRenderTarget(RHIBackend backend)
{
	return RHIRenderTarget::Create(m_device);
}

std::unique_ptr<RHIProfiler> RHIContext::CreateProfiler(RHIBackend backend)
{
	return RHIProfiler::Create(m_device, m_swapchain->GetTextureCount());
}

std::unique_ptr<RHIFence> RHIContext::CreateFence(RHIBackend backend)
{
	return RHIFence::Create(m_device);
}

std::unique_ptr<RHISemaphore> RHIContext::CreateSemaphore(RHIBackend backend)
{
	return RHISemaphore::Create(m_device);
}

std::unique_ptr<RHIAccelerationStructure> RHIContext::CreateAcccelerationStructure(RHIBackend backend)
{
	return RHIAccelerationStructure::Create(m_device);
}

RHIQueue *RHIContext::GetQueue(RHIQueueFamily family)
{
	switch (family)
	{
		case RHIQueueFamily::Graphics:
			return m_graphics_queue.get();
		case RHIQueueFamily::Compute:
			return m_compute_queue.get();
		case RHIQueueFamily::Transfer:
			return m_transfer_queue.get();
		default:
			break;
	}
	return nullptr;
}

RHIQueue *RHIContext::GetCUDAQueue()
{
	return m_cuda_queue.get();
}

std::unique_ptr<RHIQueue> RHIContext::CreateQueue(RHIQueueFamily family, uint32_t idx, RHIBackend backend)
{
	return RHIQueue::Create(m_device, family, idx);
}

RHITexture *RHIContext::GetBackBuffer()
{
	return m_swapchain->GetCurrentTexture();
}

RHITexture *RHIContext::ResourceConvert(RHITexture *src_texture, RHIBackend target)
{
	return nullptr;
}

RHIBuffer *RHIContext::ResourceConvert(RHIBuffer *src_buffer, RHIBackend target)
{
	return nullptr;
}

void RHIContext::BeginFrame()
{
	m_swapchain->AcquireNextTexture(m_present_complete[m_current_frame].get(), nullptr);
	m_frames[m_current_frame]->Reset();
}

void RHIContext::EndFrame()
{
	m_graphics_queue->Submit({}, {m_render_complete[m_current_frame].get()}, {m_present_complete[m_current_frame].get()});

	if (!m_transfer_queue->Empty())
	{
		m_transfer_queue->Execute();
		m_transfer_queue->Wait();
	}

	if (!m_compute_queue->Empty())
	{
		m_compute_queue->Execute(m_frames[m_current_frame]->AllocateFence());
	}

	if (!m_graphics_queue->Empty())
	{
		m_graphics_queue->Execute(m_frames[m_current_frame]->AllocateFence());
	}

	if (!m_swapchain->Present(m_render_complete[m_current_frame].get()) ||
	    p_window->GetWidth() != m_swapchain->GetWidth() ||
	    p_window->GetHeight() != m_swapchain->GetHeight())
	{
		m_swapchain->Resize(p_window->GetWidth(), p_window->GetHeight());
		LOG_INFO("Swapchain resize to {} x {}", p_window->GetWidth(), p_window->GetHeight());
	}

	m_current_frame = (m_current_frame + 1) % m_swapchain->GetTextureCount();
}

}        // namespace Ilum