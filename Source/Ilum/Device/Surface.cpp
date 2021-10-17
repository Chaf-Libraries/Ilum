#include "Surface.hpp"
#include "Instance.hpp"
#include "PhysicalDevice.hpp"

#include "Device/Window.hpp"
#include "Graphics/GraphicsContext.hpp"

#include "SDL_vulkan.h"

namespace Ilum
{
Surface::Surface()
{
	// Create surface handle
	SDL_Vulkan_CreateSurface(Window::instance()->getSDLHandle(), GraphicsContext::instance()->getInstance(), &m_handle);

	// Get surface capabilities
	if (!VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(GraphicsContext::instance()->getPhysicalDevice(), m_handle, &m_capabilities)))
	{
		VK_ERROR("Failed to get physical device surface capabilities!");
		return;
	}

	// Get surface format
	uint32_t surface_format_count = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(GraphicsContext::instance()->getPhysicalDevice(), m_handle, &surface_format_count, nullptr);
	std::vector<VkSurfaceFormatKHR> surface_formats(surface_format_count);
	vkGetPhysicalDeviceSurfaceFormatsKHR(GraphicsContext::instance()->getPhysicalDevice(), m_handle, &surface_format_count, surface_formats.data());

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

Surface::~Surface()
{
	if (m_handle)
	{
		vkDestroySurfaceKHR(GraphicsContext::instance()->getInstance(), m_handle, nullptr);
	}
}

Surface::operator const VkSurfaceKHR &() const
{
	return m_handle;
}

const VkSurfaceKHR &Surface::getSurface() const
{
	return m_handle;
}

const VkSurfaceCapabilitiesKHR &Surface::getCapabilities() const
{
	return m_capabilities;
}

const VkSurfaceFormatKHR &Surface::getFormat() const
{
	return m_format;
}
}        // namespace Ilum