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

class GraphicsContext : public TSubsystem<GraphicsContext>
{
  public:
	GraphicsContext(Context *context);

	const Instance &getInstance() const;

	const PhysicalDevice &getPhysicalDevice() const;

	const Surface &getSurface() const;

	const LogicalDevice &getLogicalDevice() const;

	const Swapchain &getSwapchain() const;

  private:
	scope<Instance>       m_instance;
	scope<PhysicalDevice> m_physical_device;
	scope<Surface>        m_surface;
	scope<LogicalDevice>  m_logical_device;
	scope<Swapchain>      m_swapchain;

	VkPipelineCache m_pipeline_cache = VK_NULL_HANDLE;
};
}        // namespace Ilum