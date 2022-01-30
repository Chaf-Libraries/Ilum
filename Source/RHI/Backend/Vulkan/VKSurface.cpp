#include "VKSurface.hpp"
#include "VKContext.hpp"
#include "VKInstance.hpp"

#include <Core/Window.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace Ilum::RHI::Vulkan
{
VKSurface::VKSurface(VkPhysicalDevice physical_device)
{
	VkResult result = glfwCreateWindowSurface(VKContext::GetInstance(), static_cast<GLFWwindow *>(Core::Window::GetInstance()->GetHandle()), NULL, &m_handle);

	// Get surface capabilities
	if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, m_handle, &m_capabilities) != VK_SUCCESS)
	{
		LOG_ERROR("Failed to get physical device surface capabilities!");
		return;
	}

	// Get surface format
	uint32_t surface_format_count = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, m_handle, &surface_format_count, nullptr);
	std::vector<VkSurfaceFormatKHR> surface_formats(surface_format_count);
	vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, m_handle, &surface_format_count, surface_formats.data());

	if (surface_format_count == 1 && surface_formats[0].format == VK_FORMAT_UNDEFINED)
	{
		m_format.format     = VK_FORMAT_R8G8B8A8_UNORM;
		m_format.colorSpace = surface_formats[0].colorSpace;
	}
	else
	{
		bool has_R8G8B8A8_UNORM = false;
		for (auto &surface_format : surface_formats)
		{
			if (surface_format.format == VK_FORMAT_R8G8B8A8_UNORM)
			{
				m_format           = surface_format;
				has_R8G8B8A8_UNORM = true;
				break;
			}
		}

		if (!has_R8G8B8A8_UNORM)
		{
			m_format = surface_formats[0];
		}
	}
}

VKSurface::~VKSurface()
{
	if (m_handle)
	{
		vkDestroySurfaceKHR(VKContext::GetInstance(), m_handle, nullptr);
	}
}

VKSurface::operator const VkSurfaceKHR &() const
{
	return m_handle;
}

const VkSurfaceKHR &VKSurface::GetHandle() const
{
	return m_handle;
}

const VkSurfaceCapabilitiesKHR &VKSurface::GetCapabilities() const
{
	return m_capabilities;
}

const VkSurfaceFormatKHR &VKSurface::GetFormat() const
{
	return m_format;
}
}        // namespace Ilum::RHI::Vulkan