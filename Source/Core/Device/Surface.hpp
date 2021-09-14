#pragma once

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
class Instance;
class PhysicalDevice;

class Surface
{
  public:
	Surface(const Instance& instance, const PhysicalDevice& physical_device, void *window_handle);

	~Surface();

	const VkSurfaceKHR operator&() const;

  private:
	const Instance &m_instance;
	const PhysicalDevice &m_physical_device;

	VkSurfaceKHR m_handle = VK_NULL_HANDLE;

};
}        // namespace Ilum