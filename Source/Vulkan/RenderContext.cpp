#include "RenderContext.hpp"
#include "Descriptor.hpp"
#include "Device.hpp"
#include "RenderFrame.hpp"
#include "Shader.hpp"

#include <Core/Window.hpp>

namespace Ilum::Vulkan
{
std::unique_ptr<Instance>        RenderContext::s_instance         = nullptr;
std::unique_ptr<Device>          RenderContext::s_device           = nullptr;
std::unique_ptr<ShaderCache>     RenderContext::s_shader_cache     = nullptr;
std::unique_ptr<DescriptorCache> RenderContext::s_descriptor_cache = nullptr;
std::unique_ptr<Swapchain>       RenderContext::s_swapchain        = nullptr;

std::vector<std::unique_ptr<RenderFrame>> RenderContext::s_frames;

uint32_t RenderContext::s_active_frame = 0;

RenderContext::RenderContext()
{
	s_instance         = std::make_unique<Instance>();
	s_device           = std::make_unique<Device>();
	s_shader_cache     = std::make_unique<ShaderCache>();
	s_descriptor_cache = std::make_unique<DescriptorCache>();
	s_swapchain        = std::make_unique<Swapchain>(Core::Window::GetInstance().GetWidth(), Core::Window::GetInstance().GetHeight());

	for (uint32_t i = 0; i < s_swapchain->GetImageCount(); i++)
	{
		s_frames.emplace_back(std::make_unique<RenderFrame>());
	}
}

void RenderContext::OnImGui()
{
}

void RenderContext::OnEvent(const Core::Event &event)
{
}

void RenderContext::NewFrame()
{
	s_active_frame = (s_active_frame + 1) % s_frames.size();
	GetFrame().Reset();
}

void RenderContext::EndFrame()
{
}

Instance &RenderContext::GetInstance()
{
	return *s_instance;
}

Device &RenderContext::GetDevice()
{
	return *s_device;
}

ShaderCache &RenderContext::GetShaderCache()
{
	return *s_shader_cache;
}

DescriptorCache &RenderContext::GetDescriptorCache()
{
	return *s_descriptor_cache;
}

Swapchain &RenderContext::GetSwapchain()
{
	return *s_swapchain;
}

RenderFrame &RenderContext::GetFrame()
{
	return *s_frames[s_active_frame];
}
}        // namespace Ilum::Vulkan