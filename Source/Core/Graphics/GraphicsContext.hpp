#pragma once

#include "Core/Engine/PCH.hpp"
#include "Core/Engine/Subsystem.hpp"

namespace Ilum
{
class Instance;
class PhysicalDevice;
class Surface;
class LogicalDevice;

class GraphicsContext : public TSubsystem<GraphicsContext>
{
  public:
	GraphicsContext(Context *context);

  private:
	std::unique_ptr<Instance>       m_instance;
	std::unique_ptr<PhysicalDevice> m_physical_device;
	std::unique_ptr<Surface>        m_surface;
	std::unique_ptr<LogicalDevice>  m_logical_device;

	VkPipelineCache m_pipeline_cache = VK_NULL_HANDLE;
};
}        // namespace Ilum