#include "GraphicsContext.hpp"

#include "Device/Swapchain.hpp"

#include "Engine/Context.hpp"

#include <Core/JobSystem/JobSystem.hpp>

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/Command/CommandPool.hpp"
#include "Graphics/Descriptor/DescriptorCache.hpp"
#include "Graphics/Profiler.hpp"
#include "Graphics/Shader/ShaderCache.hpp"
#include "Graphics/Synchronization/Queue.hpp"

#include <Graphics/Vulkan.hpp>
#include <Graphics/RenderContext.hpp>
#include <Graphics/Device/Device.hpp>
#include <Graphics/Device/Window.hpp>

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

const Swapchain &GraphicsContext::getSwapchain() const
{
	return *m_swapchain;
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

const ref<CommandPool> &GraphicsContext::getCommandPool(QueueUsage usage, const std::thread::id &thread_id)
{
	std::lock_guard<std::mutex> lock(m_command_pool_mutex);
	if (m_command_pools[thread_id].find(usage) == m_command_pools[thread_id].end())
	{
		return m_command_pools[thread_id].emplace(usage, createRef<CommandPool>(usage, thread_id)).first->second;
	}

	return m_command_pools[thread_id][usage];
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

const CommandBuffer &GraphicsContext::getCurrentCommandBuffer() const
{
	return *m_main_command_buffers[m_current_frame];
}

const CommandBuffer &GraphicsContext::acquireCommandBuffer(QueueUsage usage)
{
	while (m_command_buffers.find(std::this_thread::get_id()) == m_command_buffers.end() || m_command_buffers[std::this_thread::get_id()][usage].size() <= m_current_frame)
	{
		std::lock_guard<std::mutex> lock(m_command_buffer_mutex);
		m_command_buffers[std::this_thread::get_id()][usage].emplace_back(createScope<CommandBuffer>(usage));
	}

	return *m_command_buffers[std::this_thread::get_id()][usage][m_current_frame];
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

	m_command_pools.clear();
	m_main_command_buffers.clear();

	m_swapchain.reset();

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

	bool need_rebuild = m_swapchain != nullptr;

	if (need_rebuild)
	{
		VK_INFO("Recreating swapchain from ({}, {}) to ({}, {})", m_swapchain->getExtent().width, m_swapchain->getExtent().height, display_extent.width, display_extent.height);
	}

	m_swapchain = createScope<Swapchain>(display_extent, m_swapchain.get(), vsync);

	if (need_rebuild)
	{
		Swapchain_Rebuild_Event.Invoke();
	}

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

	m_flight_fences.resize(m_swapchain->getImageCount());
	m_render_complete.resize(m_swapchain->getImageCount());
	m_present_complete.resize(m_swapchain->getImageCount());
	m_main_command_buffers.resize(m_swapchain->getImageCount());

	VkSemaphoreCreateInfo semaphore_create_info = {};
	semaphore_create_info.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fence_create_info = {};
	fence_create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fence_create_info.flags             = VK_FENCE_CREATE_SIGNALED_BIT;

	for (uint32_t i = 0; i < m_swapchain->getImageCount(); i++)
	{
		vkCreateSemaphore(Graphics::RenderContext::GetDevice(), &semaphore_create_info, nullptr, &m_present_complete[i]);
		vkCreateSemaphore(Graphics::RenderContext::GetDevice(), &semaphore_create_info, nullptr, &m_render_complete[i]);
		vkCreateFence(Graphics::RenderContext::GetDevice(), &fence_create_info, nullptr, &m_flight_fences[i]);
		m_main_command_buffers[i] = createScope<CommandBuffer>(QueueUsage::Present);
		Graphics::VKDebugger::SetName(Graphics::RenderContext::GetDevice(), *m_main_command_buffers[i], ("main command buffer " + std::to_string(i)).c_str());
	}
}

void GraphicsContext::newFrame()
{
	auto acquire_result = m_swapchain->acquireNextImage(m_present_complete[m_current_frame]);
	if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR || m_swapchain->isVsync() != m_vsync)
	{
		createSwapchain(m_vsync);
		return;
	}

	if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR)
	{
		VK_ERROR("Failed to acquire swapchain image!");
		return;
	}

	vkWaitForFences(Graphics::RenderContext::GetDevice(), 1, &m_flight_fences[m_current_frame], VK_TRUE, std::numeric_limits<uint64_t>::max());
	vkResetFences(Graphics::RenderContext::GetDevice(), 1, &m_flight_fences[m_current_frame]);

	m_main_command_buffers[m_current_frame]->begin();

	m_profiler->beginFrame(*m_main_command_buffers[m_current_frame]);
}

void GraphicsContext::submitFrame()
{
	m_main_command_buffers[m_current_frame]->end();

	m_queue_system->acquire()->submit(*m_main_command_buffers[m_current_frame], m_render_complete[m_current_frame], m_present_complete[m_current_frame], m_flight_fences[m_current_frame]);
	auto &present_queue  = *m_queue_system->acquire(QueueUsage::Present, 1);
	auto  present_result = m_swapchain->present(present_queue, m_render_complete[m_current_frame]);

	if (present_result == VK_ERROR_OUT_OF_DATE_KHR || m_swapchain->isVsync() != m_vsync)
	{
		createSwapchain(m_vsync);
		return;
	}

	if (present_result != VK_SUCCESS && present_result != VK_SUBOPTIMAL_KHR)
	{
		VK_ERROR("Failed to present swapchain image!");
		return;
	}

	m_current_frame = (m_current_frame + 1) % m_swapchain->getImageCount();
}
}        // namespace Ilum