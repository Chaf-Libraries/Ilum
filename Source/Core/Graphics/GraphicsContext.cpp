#include "GraphicsContext.hpp"

#include "Core/Device/Instance.hpp"
#include "Core/Device/LogicalDevice.hpp"
#include "Core/Device/PhysicalDevice.hpp"
#include "Core/Device/Surface.hpp"
#include "Core/Device/Window.hpp"

#include "Core/Engine/Context.hpp"
#include "Core/Engine/Threading/ThreadPool.hpp"

#include "Core/Graphics/Command/CommandBuffer.hpp"
#include "Core/Graphics/Command/CommandPool.hpp"

#include "RenderPass/Swapchain.hpp"

namespace Ilum
{
GraphicsContext::GraphicsContext(Context *context) :
    TSubsystem<GraphicsContext>(context),
    m_instance(createScope<Instance>()),
    m_physical_device(createScope<PhysicalDevice>(*m_instance)),
    m_surface(createScope<Surface>(*m_instance, *m_physical_device, m_context->getSubsystem<Window>()->getSDLHandle())),
    m_logical_device(createScope<LogicalDevice>(*m_instance, *m_physical_device, *m_surface))
{
	for (auto& thread : ThreadPool::instance()->getThreads())
	{
		m_graphics_command_pools.emplace(thread.get_id(), createScope<CommandPool>(VK_QUEUE_GRAPHICS_BIT, thread.get_id()));
		m_compute_command_pools.emplace(thread.get_id(), createScope<CommandPool>(VK_QUEUE_COMPUTE_BIT, thread.get_id()));
		m_transfer_command_pools.emplace(thread.get_id(), createScope<CommandPool>(VK_QUEUE_TRANSFER_BIT, thread.get_id()));
	}

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

const CommandPool &GraphicsContext::getCommandPool(VkQueueFlagBits queue_type, const std::thread::id &thread_id) const
{
	if (queue_type & VK_QUEUE_GRAPHICS_BIT)
	{
		return *m_graphics_command_pools.at(thread_id);
	}
	else if (queue_type & VK_QUEUE_COMPUTE_BIT)
	{
		return *m_compute_command_pools.at(thread_id);
	}
	else if (queue_type & VK_QUEUE_TRANSFER_BIT)
	{
		return *m_transfer_command_pools.at(thread_id);
	}

	return *m_graphics_command_pools.at(thread_id);
}
}        // namespace Ilum