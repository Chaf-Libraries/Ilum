#include "RenderContext.hpp"
#include "RenderFrame.hpp"

#include "Device/Device.hpp"
#include "Device/Instance.hpp"
#include "Device/PhysicalDevice.hpp"
#include "Device/Surface.hpp"
#include "Device/Swapchain.hpp"
#include "Device/Window.hpp"

#include "Synchronize/FencePool.hpp"
#include "Synchronize/SemaphorePool.hpp"

namespace Ilum::Graphics
{
RenderContext::RenderContext()
{
	m_window          = std::make_unique<Window>();
	m_instance        = std::make_unique<Instance>();
	m_physical_device = std::make_unique<PhysicalDevice>(*m_instance);
	m_surface         = std::make_unique<Surface>(*m_instance, *m_physical_device, m_window->GetHandle());
	m_device          = std::make_unique<Device>(*m_instance, *m_physical_device, *m_surface);
	m_swapchain       = std::make_unique<Swapchain>(VkExtent2D{m_window->GetWidth(), m_window->GetHeight()}, *m_device, *m_surface, *m_physical_device);

	for (uint32_t i = 0; i < m_swapchain->GetImageCount(); i++)
	{
		m_render_frames.emplace_back(std::make_unique<RenderFrame>(*m_device));
	}
}

void RenderContext::NewFrame(VkSemaphore image_ready)
{
	auto result = Get().m_swapchain->AcquireNextImage(image_ready);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
	{
		Get().Recreate();
	}
	GetFrame().Reset();
}

void RenderContext::EndFrame(VkSemaphore render_complete)
{
	auto result = Get().m_swapchain->Present(render_complete);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
	{
		Get().Recreate();
	}
	Get().m_active_frame = (Get().m_active_frame + 1) % Get().m_render_frames.size();
}

void RenderContext::BeginImGui()
{
}

void RenderContext::EndImGui()
{
}

void RenderContext::SetRenderArea(const VkExtent2D &extent)
{
	Get().m_render_area = extent;
}

const VkExtent2D &RenderContext::GetRenderArea()
{
	return Get().m_render_area;
}

Window &RenderContext::GetWindow()
{
	return *Get().m_window;
}

Instance &RenderContext::GetInstance()
{
	return *Get().m_instance;
}

Surface &RenderContext::GetSurface()
{
	return *Get().m_surface;
}

Device &RenderContext::GetDevice()
{
	return *Get().m_device;
}

PhysicalDevice &RenderContext::GetPhysicalDevice()
{
	return *Get().m_physical_device;
}

Swapchain &RenderContext::GetSwapchain()
{
	return *Get().m_swapchain;
}

RenderFrame &RenderContext::GetFrame()
{
	return *Get().m_render_frames[Get().m_active_frame];
}

RenderContext &RenderContext::Get()
{
	static RenderContext render_context;
	return render_context;
}

void RenderContext::Recreate()
{
	m_device->WaitIdle();

	m_active_frame = 0;
	m_render_frames.clear();
	m_swapchain = std::make_unique<Swapchain>(VkExtent2D{m_window->GetWidth(), m_window->GetHeight()}, *m_device, *m_surface, *m_physical_device, *m_swapchain);

	for (uint32_t i = 0; i < m_swapchain->GetImageCount(); i++)
	{
		m_render_frames.emplace_back(std::make_unique<RenderFrame>(*m_device));
	}
}
}        // namespace Ilum::Graphics