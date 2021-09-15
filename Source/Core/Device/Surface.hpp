#pragma once

#include "Core/Engine/PCH.hpp"

struct SDL_Window;

namespace Ilum
{
class Instance;
class PhysicalDevice;

class Surface
{
  public:
	Surface(const Instance& instance, const PhysicalDevice& physical_device, SDL_Window *window_handle);

	~Surface();

	operator const VkSurfaceKHR &() const;

	const VkSurfaceKHR &getSurface() const;

	const VkSurfaceCapabilitiesKHR &getCapabilities() const;

	const VkSurfaceFormatKHR &getFormat() const;

  private:
	const Instance &m_instance;
	const PhysicalDevice &m_physical_device;

	VkSurfaceKHR m_handle = VK_NULL_HANDLE;
	VkSurfaceCapabilitiesKHR m_capabilities = {};
	VkSurfaceFormatKHR       m_format       = {};
};
}        // namespace Ilum