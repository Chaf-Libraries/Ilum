#include "GraphicsContext.hpp"

#include "Core/Device/Instance.hpp"
#include "Core/Device/LogicalDevice.hpp"
#include "Core/Device/PhysicalDevice.hpp"
#include "Core/Device/Surface.hpp"
#include "Core/Device/Window.hpp"

#include "Core/Engine/Context.hpp"
#include "Core/Engine/Threading/ThreadPool.hpp"

#include "Core/Graphics/Descriptor/DescriptorCache.hpp"
#include "Core/Graphics/Command/CommandBuffer.hpp"
#include "Core/Graphics/Command/CommandPool.hpp"

#include "RenderPass/Swapchain.hpp"

namespace Ilum
{
GraphicsContext::GraphicsContext(Context *context) :
    TSubsystem<GraphicsContext>(context),
    m_instance(createScope<Instance>()),
    m_physical_device(createScope<PhysicalDevice>()),
    m_surface(createScope<Surface>()),
    m_logical_device(createScope<LogicalDevice>())
{
	// Create pipeline cache
	VkPipelineCacheCreateInfo pipeline_cache_create_info = {};
	pipeline_cache_create_info.sType                     = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	vkCreatePipelineCache(*m_logical_device, &pipeline_cache_create_info, nullptr, &m_pipeline_cache);

	m_descriptor_cache = createScope<DescriptorCache>();
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

bool GraphicsContext::onInitialize()
{
	createSwapchain();
	return true;
}

void GraphicsContext::onTick(float delta_time)
{
	if (Window::instance()->isMinimized())
	{
		return;
	}

	m_current_frame = (m_current_frame + 1) % m_swapchain->getImageCount();

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

	VkExtent2D display_extent = {Window::instance()->getWidth(), Window::instance()->getHeight()};

	if (m_swapchain)
	{
		VK_INFO("Recreating swapchain from ({}, {}) to ({}, {})", m_swapchain->getExtent().width, m_swapchain->getExtent().height, display_extent.width, display_extent.height);
	}

	m_swapchain = createScope<Swapchain>(display_extent, m_swapchain.get());
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

void GraphicsContext::prepareFrame()
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
}

void GraphicsContext::submitFrame()
{
	auto &queues         = m_logical_device->getPresentQueues();
	auto &queue          = queues[m_current_frame % queues.size()];
	auto  present_result = m_swapchain->present(queue, m_render_complete[m_current_frame]);

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

	vkQueueWaitIdle(queue);
}

void GraphicsContext::draw()
{
	prepareFrame();

	// Draw call

	submitFrame();
}
}        // namespace Ilum