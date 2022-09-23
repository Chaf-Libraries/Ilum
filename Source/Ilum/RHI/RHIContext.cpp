#include "RHIContext.hpp"
#include "Backend/CUDA/Texture.hpp"
#include "Backend/Vulkan/Texture.hpp"

#include <Core/Time.hpp>

#undef CreateSemaphore

namespace Ilum
{
RHIContext::RHIContext(Window *window) :
    p_window(window)
{
	m_cuda_device = RHIDevice::Create(RHIBackend::CUDA);

#ifdef RHI_BACKEND_VULKAN
	m_device = RHIDevice::Create(RHIBackend::Vulkan);
	LOG_INFO("RHI Backend: Vulkan");
#elif RHI_BACKEND_DX12
	m_device = RHIDevice::Create(RHIBackend::DX12);
	LOG_INFO("RHI Backend: DX12");
#elif RHI_BACKEND_OPENGL
	m_device = RHIDevice::Create(RHIBackend::OpenGL);
	LOG_INFO("RHI Backend: OpenGL");
#else
#	error "Please specify a rhi backend!"
#endif        // RHI_BACKEND

	m_swapchain = RHISwapchain::Create(m_device.get(), p_window->GetNativeHandle(), p_window->GetWidth(), p_window->GetHeight(), true);

	m_queue = RHIQueue::Create(m_device.get());

	for (uint32_t i = 0; i < m_swapchain->GetTextureCount(); i++)
	{
		m_frames.emplace_back(RHIFrame::Create(m_device.get()));
		m_cuda_frames.emplace_back(RHIFrame::Create(m_cuda_device.get()));
		m_present_complete.emplace_back(RHISemaphore::Create(m_device.get()));
		m_render_complete.emplace_back(RHISemaphore::Create(m_device.get()));
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
	m_queue.reset();
	m_device.reset();
	m_cuda_device.reset();
}

RHIBackend RHIContext::GetBackend() const
{
#ifdef RHI_BACKEND_VULKAN
	return RHIBackend::Vulkan;
#elif RHI_BACKEND_DX12
	return RHIBackend::DX12;
#elif RHI_BACKEND_OPENGL
	return RHIBackend::OpenGL;
#else
#	error "Please specify a rhi backend!"
#endif        // RHI_BACKEND
}

bool RHIContext::IsFeatureSupport(RHIFeature feature) const
{
	return m_device->IsFeatureSupport(feature);
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
	return RHISwapchain::Create(m_device.get(), window_handle, width, height, sync);
}

std::unique_ptr<RHITexture> RHIContext::CreateTexture(const TextureDesc &desc, bool cuda)
{
	return RHITexture::Create(m_device.get(), desc);
}

std::unique_ptr<RHITexture> RHIContext::CreateTexture2D(uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples, bool cuda)
{
	return RHITexture::Create2D(m_device.get(), width, height, format, usage, mipmap, samples);
}

std::unique_ptr<RHITexture> RHIContext::CreateTexture3D(uint32_t width, uint32_t height, uint32_t depth, RHIFormat format, RHITextureUsage usage, bool cuda)
{
	return RHITexture::Create3D(m_device.get(), width, height, depth, format, usage);
}

std::unique_ptr<RHITexture> RHIContext::CreateTextureCube(uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap, bool cuda)
{
	return RHITexture::CreateCube(m_device.get(), width, height, format, usage, mipmap);
}

std::unique_ptr<RHITexture> RHIContext::CreateTexture2DArray(uint32_t width, uint32_t height, uint32_t layers, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples, bool cuda)
{
	return RHITexture::Create2DArray(m_device.get(), width, height, layers, format, usage, mipmap, samples);
}

std::unique_ptr<RHITexture> RHIContext::MapToCUDATexture(RHITexture *texture)
{
	return std::make_unique<CUDA::Texture>(static_cast<CUDA::Device *>(m_cuda_device.get()), static_cast<Vulkan::Device *>(m_device.get()), static_cast<Vulkan::Texture *>(texture));
}

std::unique_ptr<RHIBuffer> RHIContext::CreateBuffer(const BufferDesc &desc, bool cuda)
{
	return RHIBuffer::Create(m_device.get(), desc);
}

std::unique_ptr<RHIBuffer> RHIContext::CreateBuffer(size_t size, RHIBufferUsage usage, RHIMemoryUsage memory, bool cuda)
{
	BufferDesc desc = {};
	desc.size       = size;
	desc.usage      = usage;
	desc.memory     = memory;

	return RHIBuffer::Create(m_device.get(), desc);
}

std::unique_ptr<RHISampler> RHIContext::CreateSampler(const SamplerDesc &desc, bool cuda)
{
	return RHISampler::Create(m_device.get(), desc);
}

RHICommand *RHIContext::CreateCommand(RHIQueueFamily family, bool cuda)
{
	return m_frames[m_current_frame]->AllocateCommand(family);
}

std::unique_ptr<RHIDescriptor> RHIContext::CreateDescriptor(const ShaderMeta &meta, bool cuda)
{
	return RHIDescriptor::Create(m_device.get(), meta);
}

std::unique_ptr<RHIPipelineState> RHIContext::CreatePipelineState(bool cuda)
{
	return RHIPipelineState::Create(m_device.get());
}

std::unique_ptr<RHIShader> RHIContext::CreateShader(const std::string &entry_point, const std::vector<uint8_t> &source, bool cuda)
{
	return RHIShader::Create(cuda ? m_cuda_device.get() : m_device.get(), entry_point, source);
}

std::unique_ptr<RHIRenderTarget> RHIContext::CreateRenderTarget(bool cuda)
{
	return RHIRenderTarget::Create(m_device.get());
}

std::unique_ptr<RHIProfiler> RHIContext::CreateProfiler(bool cuda)
{
	return RHIProfiler::Create(m_device.get(), m_swapchain->GetTextureCount());
}

std::unique_ptr<RHIFence> RHIContext::CreateFence(bool cuda)
{
	return RHIFence::Create(m_device.get());
}

std::unique_ptr<RHISemaphore> RHIContext::CreateSemaphore(bool cuda)
{
	return RHISemaphore::Create(m_device.get());
}

std::unique_ptr<RHISemaphore> RHIContext::MapToCUDASemaphore(RHISemaphore *semaphore)
{
	return nullptr;
}

std::unique_ptr<RHIAccelerationStructure> RHIContext::CreateAcccelerationStructure(bool cuda)
{
	return RHIAccelerationStructure::Create(m_device.get());
}

void RHIContext::Submit(std::vector<RHICommand *> &&cmd_buffers, std::vector<RHISemaphore *> &&wait_semaphores, std::vector<RHISemaphore *> &&signal_semaphores)
{
	SubmitInfo submit_info        = {};
	submit_info.is_cuda           = cmd_buffers.empty() ? false : cmd_buffers[0]->GetBackend() == RHIBackend::CUDA;
	submit_info.queue_family      = cmd_buffers.empty() ? RHIQueueFamily::Graphics : cmd_buffers[0]->GetQueueFamily();
	submit_info.cmd_buffers       = std::move(cmd_buffers);
	submit_info.wait_semaphores   = std::move(wait_semaphores);
	submit_info.signal_semaphores = std::move(signal_semaphores);
	m_submit_infos.emplace_back(std::move(submit_info));
}

void RHIContext::Execute(RHICommand *cmd_buffer)
{
	if (cmd_buffer->GetBackend() == RHIBackend::CUDA)
	{
		m_cuda_queue->Execute(cmd_buffer);
	}
	else
	{
		m_queue->Execute(cmd_buffer);
	}
}

void RHIContext::Execute(std::vector<RHICommand *> &&cmd_buffers, std::vector<RHISemaphore *> &&wait_semaphores, std::vector<RHISemaphore *> &&signal_semaphores)
{
	SubmitInfo submit_info        = {};
	submit_info.is_cuda           = cmd_buffers.empty() ? false : cmd_buffers[0]->GetBackend() == RHIBackend::CUDA;
	submit_info.queue_family      = cmd_buffers.empty() ? RHIQueueFamily::Graphics : cmd_buffers[0]->GetQueueFamily();
	submit_info.cmd_buffers       = std::move(cmd_buffers);
	submit_info.wait_semaphores   = std::move(wait_semaphores);
	submit_info.signal_semaphores = std::move(signal_semaphores);

	if (submit_info.is_cuda)
	{
		m_cuda_queue->Execute(submit_info.queue_family, {submit_info});
	}
	else
	{
		m_queue->Execute(submit_info.queue_family, {submit_info});
	}
}

void RHIContext::Reset()
{
	m_submit_infos.clear();
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
	if (!m_submit_infos.empty())
	{
		for (int32_t i = static_cast<int32_t>(m_submit_infos.size()) - 1; i >= 0; i--)
		{
			if (!m_submit_infos[i].is_cuda)
			{
				m_submit_infos[i].signal_semaphores.push_back(m_render_complete[m_current_frame].get());
				m_submit_infos[i].wait_semaphores.push_back(m_present_complete[m_current_frame].get());
				break;
			}
		}

		std::vector<SubmitInfo> pack_submit_infos;
		pack_submit_infos.reserve(m_submit_infos.size());
		RHIQueueFamily last_queue_family = m_submit_infos[0].queue_family;
		bool           last_is_cuda      = m_submit_infos[0].is_cuda;

		for (auto &submit_info : m_submit_infos)
		{
			if (last_queue_family != submit_info.queue_family || last_is_cuda != submit_info.is_cuda)
			{
				if (last_is_cuda)
				{
					m_cuda_queue->Execute(last_queue_family, pack_submit_infos);
				}
				else
				{
					m_queue->Execute(last_queue_family, pack_submit_infos, m_frames[m_current_frame]->AllocateFence());
				}
				pack_submit_infos.clear();
				last_queue_family = submit_info.queue_family;
			}
			pack_submit_infos.push_back(submit_info);
		}
		if (!pack_submit_infos.empty())
		{
			m_queue->Execute(last_queue_family, pack_submit_infos, m_frames[m_current_frame]->AllocateFence());
		}
		m_submit_infos.clear();
	}

	/*m_graphics_queue->Submit({}, {m_render_complete[m_current_frame].get()}, {m_present_complete[m_current_frame].get()});

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
	}*/

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