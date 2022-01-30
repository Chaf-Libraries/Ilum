#pragma once

#include "Vulkan.hpp"

namespace Ilum::RHI::Vulkan
{
class VKSurface
{
  public:
	VKSurface(VkPhysicalDevice physical_device);

	~VKSurface();

	operator const VkSurfaceKHR &() const;

	const VkSurfaceKHR &GetHandle() const;

	const VkSurfaceCapabilitiesKHR &GetCapabilities() const;

	const VkSurfaceFormatKHR &GetFormat() const;

  private:
	VkSurfaceKHR             m_handle       = VK_NULL_HANDLE;
	VkSurfaceCapabilitiesKHR m_capabilities = {};
	VkSurfaceFormatKHR       m_format       = {};
};
}        // namespace Ilum::RHI::Vulkan