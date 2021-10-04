#pragma once

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
class Surface
{
  public:
	Surface();

	~Surface();

	operator const VkSurfaceKHR &() const;

	const VkSurfaceKHR &getSurface() const;

	const VkSurfaceCapabilitiesKHR &getCapabilities() const;

	const VkSurfaceFormatKHR &getFormat() const;

  private:
	VkSurfaceKHR m_handle = VK_NULL_HANDLE;
	VkSurfaceCapabilitiesKHR m_capabilities = {};
	VkSurfaceFormatKHR       m_format       = {};
};
}        // namespace Ilum