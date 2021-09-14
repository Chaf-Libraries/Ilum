#include "Surface.hpp"
#include "Instance.hpp"
#include "PhysicalDevice.hpp"

namespace Ilum
{
Surface::Surface(const Instance &instance, const PhysicalDevice &physical_device, void *window_handle):
    m_instance(instance),
    m_physical_device(physical_device)
{
}

Surface::~Surface()
{
	if (m_handle)
	{
		vkDestroySurfaceKHR(m_instance, m_handle, nullptr);
	}
}

const VkSurfaceKHR Surface::operator&() const
{
	return VkSurfaceKHR();
}
}        // namespace Ilum