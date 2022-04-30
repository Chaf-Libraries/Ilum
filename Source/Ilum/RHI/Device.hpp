#pragma once

#include <vk_mem_alloc.h>
#include <volk.h>

#include <vector>

namespace Ilum
{
class Window;

class RHIDevice
{
  public:
	RHIDevice(Window *window);

	~RHIDevice();

  private:

  private:
	void CreateInstance();
	void CreatePhysicalDevice();

  private:
	// Extension functions
	static PFN_vkCreateDebugUtilsMessengerEXT          vkCreateDebugUtilsMessengerEXT;
	static VkDebugUtilsMessengerEXT                    vkDebugUtilsMessengerEXT;
	static PFN_vkDestroyDebugUtilsMessengerEXT         vkDestroyDebugUtilsMessengerEXT;
	static PFN_vkSetDebugUtilsObjectTagEXT             vkSetDebugUtilsObjectTagEXT;
	static PFN_vkSetDebugUtilsObjectNameEXT            vkSetDebugUtilsObjectNameEXT;
	static PFN_vkCmdBeginDebugUtilsLabelEXT            vkCmdBeginDebugUtilsLabelEXT;
	static PFN_vkCmdEndDebugUtilsLabelEXT              vkCmdEndDebugUtilsLabelEXT;
	static PFN_vkGetPhysicalDeviceMemoryProperties2KHR vkGetPhysicalDeviceMemoryProperties2KHR;

  private:
	static const std::vector<const char *>                 s_instance_extensions;
	static const std::vector<const char *>                 s_validation_layers;
	static const std::vector<VkValidationFeatureEnableEXT> s_validation_extensions;

  private:
	VkInstance       m_instance;
	VkPhysicalDevice m_physical_device;
	VkDevice         m_device;
};
}        // namespace Ilum