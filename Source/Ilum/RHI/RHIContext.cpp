#include "RHIContext.hpp"

namespace Ilum
{
RHIContext::RHIContext(Window *window) :
    p_window(window)
{
	m_device    = RHIDevice::Create();
	m_swapchain = RHISwapchain::Create(m_device.get(), p_window);

	m_graphics_queue = RHIQueue::Create(m_device.get(), RHIQueueFamily::Graphics);
	m_compute_queue  = RHIQueue::Create(m_device.get(), RHIQueueFamily::Compute);
	m_transfer_queue = RHIQueue::Create(m_device.get(), RHIQueueFamily::Transfer);

	for (uint32_t i = 0; i < m_swapchain->GetTextureCount(); i++)
	{
		m_present_complete.emplace_back(RHISemaphore::Create(m_device.get()));
		m_render_complete.emplace_back(RHISemaphore::Create(m_device.get()));
	}
}

RHIContext::~RHIContext()
{
	m_device->WaitIdle();

	m_present_complete.clear();
	m_render_complete.clear();

	m_swapchain.reset();
	m_device.reset();
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

void RHIContext::BeginFrame()
{
	uint32_t current_index = m_swapchain->GetCurrentFrameIndex();
	uint32_t texture_count = m_swapchain->GetTextureCount();

	m_swapchain->AcquireNextTexture(m_present_complete[(current_index + 1) % texture_count].get(), nullptr);
}

void RHIContext::EndFrame()
{
	m_transfer_queue->Execute();
	m_compute_queue->Execute();
	m_graphics_queue->Execute();

	m_swapchain->Present(m_present_complete[m_swapchain->GetCurrentFrameIndex()].get());
}
}        // namespace Ilum