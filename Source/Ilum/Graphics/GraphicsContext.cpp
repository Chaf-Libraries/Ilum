#include "GraphicsContext.hpp"

#include "Device/Instance.hpp"
#include "Device/LogicalDevice.hpp"
#include "Device/PhysicalDevice.hpp"
#include "Device/Surface.hpp"
#include "Device/Swapchain.hpp"
#include "Device/Window.hpp"

#include "Engine/Context.hpp"

#include "Threading/ThreadPool.hpp"

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/Command/CommandPool.hpp"
#include "Graphics/Descriptor/DescriptorCache.hpp"
#include "Graphics/Profiler.hpp"
#include "Graphics/RenderFrame.hpp"
#include "Graphics/Shader/ShaderCache.hpp"
#include "Graphics/Synchronization/FencePool.hpp"
#include "Graphics/Synchronization/Queue.hpp"
#include "Graphics/Synchronization/SemaphorePool.hpp"
#include "Graphics/Vulkan/VK_Debugger.h"

#include "ImGui/ImGuiContext.hpp"

#include <imgui.h>

namespace Ilum
{
GraphicsContext::GraphicsContext(Context *context) :
    TSubsystem<GraphicsContext>(context),
    m_instance(createScope<Instance>()),
    m_physical_device(createScope<PhysicalDevice>()),
    m_surface(createScope<Surface>()),
    m_logical_device(createScope<LogicalDevice>()),
    m_descriptor_cache(createScope<DescriptorCache>()),
    m_shader_cache(createScope<ShaderCache>()),
    m_queue_system(createScope<QueueSystem>())
{
	// Create pipeline cache
	VkPipelineCacheCreateInfo pipeline_cache_create_info = {};
	pipeline_cache_create_info.sType                     = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	vkCreatePipelineCache(*m_logical_device, &pipeline_cache_create_info, nullptr, &m_pipeline_cache);
	createSwapchain();
}

const Instance &GraphicsContext::getInstance() const
{
	return *m_instance;
}

const PhysicalDevice &GraphicsContext::getPhysicalDevice() const
{
	return *m_physical_device;
}

const Surface &GraphicsContext::getSurface() const
{
	return *m_surface;
}

const LogicalDevice &GraphicsContext::getLogicalDevice() const
{
	return *m_logical_device;
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

CommandPool &GraphicsContext::getCommandPool(QueueUsage usage, CommandPool::ResetMode reset_mode)
{
	return m_render_frames[m_current_frame]->requestCommandPool(usage, reset_mode);
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

void GraphicsContext::submitCommandBuffer(VkCommandBuffer cmd_buffer)
{
	m_submit_cmd_buffers.push_back(cmd_buffer);
}

RenderFrame &GraphicsContext::getFrame()
{
	return *m_render_frames[m_current_frame];
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
	if (Window::instance()->isMinimized())
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
	ThreadPool::instance()->waitAll();

	GraphicsContext::instance()->getQueueSystem().waitAll();

	for (uint32_t i = 0; i < m_render_complete.size(); i++)
	{
		vkDestroySemaphore(*m_logical_device, m_render_complete[i], nullptr);
		vkDestroySemaphore(*m_logical_device, m_present_complete[i], nullptr);
	}

	m_render_frames.clear();

	m_swapchain.reset();

	vkDestroyPipelineCache(*m_logical_device, m_pipeline_cache, nullptr);
}

void GraphicsContext::createSwapchain(bool vsync)
{
	GraphicsContext::instance()->getQueueSystem().waitAll();

	m_current_frame = 0;

	VkExtent2D display_extent = {Window::instance()->getWidth(), Window::instance()->getHeight()};

	while (Window::instance()->isMinimized())
	{
		Window::instance()->pollEvent();
	}

	bool need_rebuild = m_swapchain != nullptr;

	if (need_rebuild)
	{
		VK_INFO("Recreating swapchain from ({}, {}) to ({}, {})", m_swapchain->getExtent().width, m_swapchain->getExtent().height, display_extent.width, display_extent.height);
	}

	m_swapchain = createScope<Swapchain>(display_extent, m_swapchain.get(), vsync);

	if (need_rebuild)
	{
		Swapchain_Rebuild_Event.invoke();
	}

	for (uint32_t i = 0; i < m_render_complete.size(); i++)
	{
		vkDestroySemaphore(*m_logical_device, m_render_complete[i], nullptr);
		vkDestroySemaphore(*m_logical_device, m_present_complete[i], nullptr);
	}

	m_render_frames.clear();
	for (uint32_t i = 0; i < m_swapchain->getImageCount(); i++)
	{
		m_render_frames.emplace_back(createScope<RenderFrame>());
	}

	m_render_complete.resize(m_swapchain->getImageCount());
	m_present_complete.resize(m_swapchain->getImageCount());

	VkSemaphoreCreateInfo semaphore_create_info = {};
	semaphore_create_info.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fence_create_info = {};
	fence_create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fence_create_info.flags             = VK_FENCE_CREATE_SIGNALED_BIT;

	for (uint32_t i = 0; i < m_swapchain->getImageCount(); i++)
	{
		vkCreateSemaphore(*m_logical_device, &semaphore_create_info, nullptr, &m_present_complete[i]);
		vkCreateSemaphore(*m_logical_device, &semaphore_create_info, nullptr, &m_render_complete[i]);
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

	m_render_frames[m_current_frame]->reset();
	m_submit_cmd_buffers.clear();
}

void GraphicsContext::submitFrame()
{
	m_queue_system->acquire()->submit(m_submit_cmd_buffers, m_render_complete[m_current_frame], m_present_complete[m_current_frame], m_render_frames[m_current_frame]->requestFence());
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