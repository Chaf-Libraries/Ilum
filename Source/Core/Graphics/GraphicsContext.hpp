#pragma once

#include "Core/Engine/PCH.hpp"
#include "Core/Engine/Subsystem.hpp"

namespace Ilum
{
class Instance;
class PhysicalDevice;
class Surface;
class LogicalDevice;
class Swapchain;
class CommandBuffer;
class CommandPool;

class GraphicsContext : public TSubsystem<GraphicsContext>
{
  public:
	GraphicsContext(Context *context);

	~GraphicsContext() = default;

	const Instance &getInstance() const;

	const PhysicalDevice &getPhysicalDevice() const;

	const Surface &getSurface() const;

	const LogicalDevice &getLogicalDevice() const;

	const Swapchain &getSwapchain() const;

	const CommandPool &getCommandPool(VkQueueFlagBits queue_type = VK_QUEUE_GRAPHICS_BIT, const std::thread::id &thread_id = std::this_thread::get_id()) const;

  private:
	void resizeSwapchain(uint32_t width, uint32_t height);

  private:
	scope<Instance>       m_instance;
	scope<PhysicalDevice> m_physical_device;
	scope<Surface>        m_surface;
	scope<LogicalDevice>  m_logical_device;
	scope<Swapchain>      m_swapchain;

	// Command pool per thread
	std::unordered_map<std::thread::id, scope<CommandPool>> m_graphics_command_pools;
	std::unordered_map<std::thread::id, scope<CommandPool>> m_compute_command_pools;
	std::unordered_map<std::thread::id, scope<CommandPool>> m_transfer_command_pools;

	// Present command
	scope<CommandPool>                m_present_command_pool = nullptr;
	std::vector<scope<CommandBuffer>> m_command_buffers;

	VkPipelineCache m_pipeline_cache = VK_NULL_HANDLE;
};
}        // namespace Ilum