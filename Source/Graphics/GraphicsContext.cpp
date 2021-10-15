#include "GraphicsContext.hpp"

#include "Device/Instance.hpp"
#include "Device/LogicalDevice.hpp"
#include "Device/PhysicalDevice.hpp"
#include "Device/Surface.hpp"
#include "Device/Window.hpp"
#include "Device/Swapchain.hpp"

#include "Engine/Context.hpp"

#include "Threading/ThreadPool.hpp"

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/Command/CommandPool.hpp"
#include "Graphics/Descriptor/DescriptorCache.hpp"

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
    m_descriptor_cache(createScope<DescriptorCache>())
{
	// Create pipeline cache
	VkPipelineCacheCreateInfo pipeline_cache_create_info = {};
	pipeline_cache_create_info.sType                     = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	vkCreatePipelineCache(*m_logical_device, &pipeline_cache_create_info, nullptr, &m_pipeline_cache);
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

const VkPipelineCache &GraphicsContext::getPipelineCache() const
{
	return m_pipeline_cache;
}

const ref<CommandPool> &GraphicsContext::getCommandPool(VkQueueFlagBits queue_type, const std::thread::id &thread_id)
{
	auto select_type = queue_type & VK_QUEUE_GRAPHICS_BIT ? VK_QUEUE_GRAPHICS_BIT :
                                                            (queue_type & VK_QUEUE_COMPUTE_BIT ? VK_QUEUE_COMPUTE_BIT :
                                                                                                 (queue_type & VK_QUEUE_TRANSFER_BIT ? VK_QUEUE_TRANSFER_BIT :
                                                                                                                                       VK_QUEUE_GRAPHICS_BIT));

	if (m_command_pools.find(thread_id) == m_command_pools.end())
	{
		m_command_pools[thread_id] = {};
	}

	if (m_command_pools[thread_id].find(select_type) == m_command_pools[thread_id].end())
	{
		return m_command_pools[thread_id].emplace(select_type, createRef<CommandPool>(select_type, thread_id)).first->second;
	}

	return m_command_pools[thread_id][select_type];
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
	return *m_command_buffers[m_current_frame];
}

bool GraphicsContext::onInitialize()
{
	createSwapchain();

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

	// Erase unused command pools
	if (m_stopwatch.elapsedSecond() > 10.0)
	{
		for (auto &subpool = m_command_pools.begin(); subpool != m_command_pools.end();)
		{
			for (auto &it = subpool->second.begin(); it != subpool->second.end();)
			{
				if ((*it).second.use_count() <= 1)
				{
					it = subpool->second.erase(it);
					continue;
				}
				it++;
			}

			if (subpool->second.empty())
			{
				subpool = m_command_pools.erase(subpool);
				continue;
			}
			subpool++;
		}
	}

	m_frame_count++;
}

void GraphicsContext::onPostTick()
{
	m_command_buffers[m_current_frame]->end();
	submitFrame();
}

void GraphicsContext::onShutdown()
{
	for (uint32_t i = 0; i < m_flight_fences.size(); i++)
	{
		vkDestroyFence(*m_logical_device, m_flight_fences[i], nullptr);
		vkDestroySemaphore(*m_logical_device, m_render_complete[i], nullptr);
		vkDestroySemaphore(*m_logical_device, m_present_complete[i], nullptr);
	}

	m_command_buffers.clear();
	m_command_pools.clear();

	vkDestroyPipelineCache(*m_logical_device, m_pipeline_cache, nullptr);
}

void GraphicsContext::createSwapchain()
{
	vkDeviceWaitIdle(*m_logical_device);

	m_current_frame = 0;

	VkExtent2D display_extent = {Window::instance()->getWidth(), Window::instance()->getHeight()};

	bool need_rebuild = m_swapchain != nullptr;

	if (need_rebuild)
	{
		VK_INFO("Recreating swapchain from ({}, {}) to ({}, {})", m_swapchain->getExtent().width, m_swapchain->getExtent().height, display_extent.width, display_extent.height);
	}

	m_swapchain = createScope<Swapchain>(display_extent, m_swapchain.get());

	if (need_rebuild)
	{
		Swapchain_Rebuild_Event.invoke();
	}

	createCommandBuffer();
}

void GraphicsContext::createCommandBuffer()
{
	for (uint32_t i = 0; i < m_flight_fences.size(); i++)
	{
		vkDestroyFence(*m_logical_device, m_flight_fences[i], nullptr);
		vkDestroySemaphore(*m_logical_device, m_render_complete[i], nullptr);
		vkDestroySemaphore(*m_logical_device, m_present_complete[i], nullptr);
	}

	m_flight_fences.resize(m_swapchain->getImageCount());
	m_render_complete.resize(m_swapchain->getImageCount());
	m_present_complete.resize(m_swapchain->getImageCount());
	m_command_buffers.resize(m_swapchain->getImageCount());

	VkSemaphoreCreateInfo semaphore_create_info = {};
	semaphore_create_info.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fence_create_info = {};
	fence_create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fence_create_info.flags             = VK_FENCE_CREATE_SIGNALED_BIT;

	for (uint32_t i = 0; i < m_swapchain->getImageCount(); i++)
	{
		vkCreateSemaphore(*m_logical_device, &semaphore_create_info, nullptr, &m_present_complete[i]);
		vkCreateSemaphore(*m_logical_device, &semaphore_create_info, nullptr, &m_render_complete[i]);
		vkCreateFence(*m_logical_device, &fence_create_info, nullptr, &m_flight_fences[i]);
		m_command_buffers[i] = createScope<CommandBuffer>(m_logical_device->getPresentQueueFlag());
	}
}

void GraphicsContext::newFrame()
{
	auto acquire_result = m_swapchain->acquireNextImage(m_present_complete[m_current_frame], m_flight_fences[m_current_frame]);
	if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		createSwapchain();
		return;
	}

	if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR)
	{
		VK_ERROR("Failed to acquire swapchain image!");
		return;
	}

	m_command_buffers[m_current_frame]->begin();
}

void GraphicsContext::submitFrame()
{
	m_command_buffers[m_current_frame]->submit(m_present_complete[m_current_frame], m_render_complete[m_current_frame]); 

	auto present_result = m_swapchain->present(m_logical_device->getPresentQueues()[0], m_render_complete[m_current_frame]);

	if (present_result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		createSwapchain();
		return;
	}

	if (present_result != VK_SUCCESS && present_result != VK_SUBOPTIMAL_KHR)
	{
		VK_ERROR("Failed to present swapchain image!");
		return;
	}

	vkQueueWaitIdle(m_logical_device->getPresentQueues()[0]);

	m_current_frame = (m_current_frame + 1) % m_swapchain->getImageCount();
}
}        // namespace Ilum