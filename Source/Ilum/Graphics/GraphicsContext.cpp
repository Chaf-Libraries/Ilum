#include "GraphicsContext.hpp"

#include "Engine/Context.hpp"

#include <Core/JobSystem/JobSystem.hpp>

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/Command/CommandPool.hpp"
#include "Graphics/Descriptor/DescriptorCache.hpp"
#include "Graphics/Profiler.hpp"
#include "Graphics/Shader/ShaderCache.hpp"
#include "Graphics/Synchronization/Queue.hpp"

#include <Graphics/Device/Device.hpp>
#include <Graphics/Device/Swapchain.hpp>
#include <Graphics/Device/Window.hpp>
#include <Graphics/RenderContext.hpp>
#include <Graphics/Vulkan.hpp>

#include "ImGui/ImGuiContext.hpp"

#include <imgui.h>

namespace Ilum
{
GraphicsContext::GraphicsContext(Context *context) :
    TSubsystem<GraphicsContext>(context),
    m_descriptor_cache(createScope<DescriptorCache>()),
    m_shader_cache(createScope<ShaderCache>()),
    m_queue_system(createScope<QueueSystem>())
{
	// Create pipeline cache
	VkPipelineCacheCreateInfo pipeline_cache_create_info = {};
	pipeline_cache_create_info.sType                     = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	vkCreatePipelineCache(Graphics::RenderContext::GetDevice(), &pipeline_cache_create_info, nullptr, &m_pipeline_cache);
}

DescriptorCache &GraphicsContext::getDescriptorCache()
{
	return *m_descriptor_cache;
}

ShaderCache &GraphicsContext::getShaderCache()
{
	return *m_shader_cache;
}

QueueSystem &GraphicsContext::getQueueSystem()
{
	return *m_queue_system;
}

Profiler &GraphicsContext::getProfiler()
{
	return *m_profiler;
}

const VkPipelineCache &GraphicsContext::getPipelineCache() const
{
	return m_pipeline_cache;
}

uint32_t GraphicsContext::getFrameIndex() const
{
	return m_current_frame;
}

const VkSemaphore &GraphicsContext::getPresentCompleteSemaphore() const
{
	return m_present_complete[m_current_frame];
}

const VkSemaphore &GraphicsContext::getRenderCompleteSemaphore() const
{
	return m_render_complete[m_current_frame];
}

uint64_t GraphicsContext::getFrameCount() const
{
	return m_frame_count;
}

bool GraphicsContext::isVsync() const
{
	return m_vsync;
}

void GraphicsContext::setVsync(bool vsync)
{
	m_vsync = vsync;
}

Graphics::CommandBuffer* GraphicsContext::getCurrentCommandBuffer() const
{
	return m_cmd_buffer;
}

bool GraphicsContext::onInitialize()
{
	createSwapchain(m_vsync);

	m_profiler = createScope<Profiler>();

	return true;
}

void GraphicsContext::onPreTick()
{
	newFrame();
}

void GraphicsContext::onTick(float delta_time)
{
	if (Graphics::RenderContext::GetWindow().IsMinimized())
	{
		return;
	}

	m_frame_count++;
}

void GraphicsContext::onPostTick()
{
	submitFrame();
}

void GraphicsContext::onShutdown()
{
	GraphicsContext::instance()->getQueueSystem().waitAll();

	for (uint32_t i = 0; i < m_flight_fences.size(); i++)
	{
		vkDestroyFence(Graphics::RenderContext::GetDevice(), m_flight_fences[i], nullptr);
		vkDestroySemaphore(Graphics::RenderContext::GetDevice(), m_render_complete[i], nullptr);
		vkDestroySemaphore(Graphics::RenderContext::GetDevice(), m_present_complete[i], nullptr);
	}

	vkDestroyPipelineCache(Graphics::RenderContext::GetDevice(), m_pipeline_cache, nullptr);
}

void GraphicsContext::createSwapchain(bool vsync)
{
	GraphicsContext::instance()->getQueueSystem().waitAll();

	m_current_frame = 0;

	VkExtent2D display_extent = {Graphics::RenderContext::GetWindow().GetWidth(), Graphics::RenderContext::GetWindow().GetHeight()};

	while (Graphics::RenderContext::GetWindow().IsMinimized())
	{
		Graphics::RenderContext::GetWindow().PollEvent();
	}

	VK_INFO("Recreating swapchain from ({}, {}) to ({}, {})", Graphics::RenderContext::GetSwapchain().GetExtent().width, Graphics::RenderContext::GetSwapchain().GetExtent().height, display_extent.width, display_extent.height);

	Swapchain_Rebuild_Event.Invoke();

	createCommandBuffer();
}

void GraphicsContext::createCommandBuffer()
{
	for (uint32_t i = 0; i < m_flight_fences.size(); i++)
	{
		vkDestroyFence(Graphics::RenderContext::GetDevice(), m_flight_fences[i], nullptr);
		vkDestroySemaphore(Graphics::RenderContext::GetDevice(), m_render_complete[i], nullptr);
		vkDestroySemaphore(Graphics::RenderContext::GetDevice(), m_present_complete[i], nullptr);
	}

	m_flight_fences.resize(Graphics::RenderContext::GetSwapchain().GetImageCount());
	m_render_complete.resize(Graphics::RenderContext::GetSwapchain().GetImageCount());
	m_present_complete.resize(Graphics::RenderContext::GetSwapchain().GetImageCount());

	VkSemaphoreCreateInfo semaphore_create_info = {};
	semaphore_create_info.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fence_create_info = {};
	fence_create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fence_create_info.flags             = VK_FENCE_CREATE_SIGNALED_BIT;

	for (uint32_t i = 0; i < Graphics::RenderContext::GetSwapchain().GetImageCount(); i++)
	{
		vkCreateSemaphore(Graphics::RenderContext::GetDevice(), &semaphore_create_info, nullptr, &m_present_complete[i]);
		vkCreateSemaphore(Graphics::RenderContext::GetDevice(), &semaphore_create_info, nullptr, &m_render_complete[i]);
		vkCreateFence(Graphics::RenderContext::GetDevice(), &fence_create_info, nullptr, &m_flight_fences[i]);
	}
}

void GraphicsContext::newFrame()
{
	vkWaitForFences(Graphics::RenderContext::GetDevice(), 1, &m_flight_fences[m_current_frame], VK_TRUE, std::numeric_limits<uint64_t>::max());
	vkResetFences(Graphics::RenderContext::GetDevice(), 1, &m_flight_fences[m_current_frame]);

	Graphics::RenderContext::NewFrame(m_present_complete[m_current_frame]);
	//auto acquire_result = Graphics::RenderContext::GetSwapchain().AcquireNextImage(m_present_complete[m_current_frame]);

	//if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR || acquire_result == VK_SUBOPTIMAL_KHR)
	//{
	//	Graphics::RenderContext::Get().Recreate();
	//	createSwapchain();
	//}

	//if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR)
	//{
	//	VK_ERROR("Failed to acquire swapchain image!");
	//	return;
	//}

	m_cmd_buffer = &Graphics::RenderContext::GetFrame().RequestCommandBuffer();
	m_cmd_buffer->Begin();
	Graphics::RenderContext::SetName(*m_cmd_buffer, (std::string("main command buffer ")+std::to_string(m_current_frame)).c_str());

	m_profiler->beginFrame(*m_cmd_buffer);
}

void GraphicsContext::submitFrame()
{
	m_cmd_buffer->End();

	m_queue_system->acquire()->submit(*m_cmd_buffer, m_render_complete[m_current_frame], m_present_complete[m_current_frame], m_flight_fences[m_current_frame]);
	auto &present_queue = *m_queue_system->acquire(QueueUsage::Present, 1);

	/*auto present_result = Graphics::RenderContext::GetSwapchain().Present(m_render_complete[m_current_frame]);

	if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR)
	{
		Graphics::RenderContext::Get().Recreate();
		createSwapchain();
	}
	else if (present_result != VK_SUCCESS && present_result != VK_SUBOPTIMAL_KHR)
	{
		VK_ERROR("Failed to present swapchain image!");
		return;
	}*/
	Graphics::RenderContext::EndFrame(m_render_complete[m_current_frame]);

	m_current_frame = (m_current_frame + 1) % Graphics::RenderContext::GetSwapchain().GetImageCount();
}
}        // namespace Ilum