#include "RHIContext.hpp"

namespace Ilum
{
RHIContext::RHIContext(Window *window) :
    p_window(window)
{
#ifdef RHI_BACKEND_VULKAN
	LOG_INFO("RHI Backend: Vulkan");
#elif RHI_BACKEND_DX12
	LOG_INFO("RHI Backend: DX12");
#endif        // RHI_BACKEND

	m_device    = RHIDevice::Create();
	m_swapchain = RHISwapchain::Create(m_device.get(), p_window);

	m_graphics_queue = RHIQueue::Create(m_device.get(), RHIQueueFamily::Graphics);
	m_compute_queue  = RHIQueue::Create(m_device.get(), RHIQueueFamily::Compute);
	m_transfer_queue = RHIQueue::Create(m_device.get(), RHIQueueFamily::Transfer);

	for (uint32_t i = 0; i < m_swapchain->GetTextureCount(); i++)
	{
		m_present_complete.emplace_back(RHISemaphore::Create(m_device.get()));
		m_render_complete.emplace_back(RHISemaphore::Create(m_device.get()));
		m_inflight_fence.emplace_back(RHIFence::Create(m_device.get()));
	}
}

RHIContext::~RHIContext()
{
	m_device->WaitIdle();

	for (auto &fence : m_inflight_fence)
	{
		fence->Wait();
	}

	m_present_complete.clear();
	m_render_complete.clear();
	m_inflight_fence.clear();

	m_cmds.clear();

	m_swapchain.reset();
	m_device.reset();
}

RHIBackend RHIContext::GetBackend() const
{
#ifdef RHI_BACKEND_VULKAN
	return RHIBackend::Vulkan;
#elif RHI_BACKEND_DX12
	return RHIBackend::DX12;
#endif        // RHI_BACKEND

	return RHIBackend::Unknown;
}

std::unique_ptr<RHITexture> RHIContext::CreateTexture(const TextureDesc &desc)
{
	return RHITexture::Create(m_device.get(), desc);
}

std::unique_ptr<RHITexture> RHIContext::CreateTexture2D(uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples)
{
	return RHITexture::Create2D(m_device.get(), width, height, format, usage, mipmap, samples);
}

std::unique_ptr<RHITexture> RHIContext::CreateTexture3D(uint32_t width, uint32_t height, uint32_t depth, RHIFormat format, RHITextureUsage usage)
{
	return RHITexture::Create3D(m_device.get(), width, height, depth, format, usage);
}

std::unique_ptr<RHITexture> RHIContext::CreateTextureCube(uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap)
{
	return RHITexture::CreateCube(m_device.get(), width, height, format, usage, mipmap);
}

std::unique_ptr<RHITexture> RHIContext::CreateTexture2DArray(uint32_t width, uint32_t height, uint32_t layers, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples)
{
	return RHITexture::Create2DArray(m_device.get(), width, height, layers, format, usage, mipmap, samples);
}

std::unique_ptr<RHIBuffer> RHIContext::CreateBuffer(const BufferDesc &desc)
{
	return RHIBuffer::Create(m_device.get(), desc);
}

std::unique_ptr<RHISampler> RHIContext::CreateSampler(const SamplerDesc &desc)
{
	return RHISampler::Create(m_device.get(), desc);
}

RHICommand *RHIContext::CreateCommand(RHIQueueFamily family)
{
	size_t hash = 0;
	HashCombine(hash, m_swapchain->GetCurrentFrameIndex(), family, std::this_thread::get_id());

	if (m_cmds.find(hash) != m_cmds.end())
	{
		for (auto &cmd : m_cmds[hash])
		{
			if (cmd->GetState() == CommandState::Available)
			{
				cmd->Init();
				return cmd.get();
			}
		}
	}

	m_cmds[hash].emplace_back(RHICommand::Create(m_device.get(), m_swapchain->GetCurrentFrameIndex(), family));
	m_cmds[hash].back()->Init();
	return m_cmds[hash].back().get();
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

RHITexture *RHIContext::GetBackBuffer()
{
	return m_swapchain->GetCurrentTexture();
}

void RHIContext::BeginFrame()
{
	m_inflight_fence[m_current_frame]->Wait();
	m_swapchain->AcquireNextTexture(m_present_complete[m_current_frame].get(), nullptr);
	m_inflight_fence[m_current_frame]->Reset();
}

void RHIContext::EndFrame()
{
	m_graphics_queue->Submit({}, {m_render_complete[m_current_frame].get()}, {m_present_complete[m_current_frame].get()});

	m_transfer_queue->Execute();
	m_compute_queue->Execute();
	m_graphics_queue->Execute(m_inflight_fence[m_current_frame].get());

	m_swapchain->Present(m_render_complete[m_current_frame].get());

	m_current_frame = (m_current_frame + 1) % m_swapchain->GetTextureCount();
}

}        // namespace Ilum