#pragma once

#include "Core/Engine/PCH.hpp"

#include "Core/Engine/Vulkan/Vulkan.hpp"

namespace Ilum
{
class Instance
{
  public:
	Instance();

	~Instance();

	operator const VkInstance &() const;

	const VkInstance &getInstance() const;

  public:
	// Extension functions
	static PFN_vkCreateDebugUtilsMessengerEXT          createDebugUtilsMessengerEXT;
	static VkDebugUtilsMessengerEXT                    debugUtilsMessengerEXT;
	static PFN_vkDestroyDebugUtilsMessengerEXT         destroyDebugUtilsMessengerEXT;
	static PFN_vkSetDebugUtilsObjectTagEXT             setDebugUtilsObjectTagEXT;
	static PFN_vkSetDebugUtilsObjectNameEXT            setDebugUtilsObjectNameEXT;
	static PFN_vkCmdBeginDebugUtilsLabelEXT            cmdBeginDebugUtilsLabelEXT;
	static PFN_vkCmdEndDebugUtilsLabelEXT              cmdEndDebugUtilsLabelEXT;
	static PFN_vkGetPhysicalDeviceMemoryProperties2KHR getPhysicalDeviceMemoryProperties2KHR;

  public:
	// Extensions
	static const std::vector<const char *>                 extensions;
	static const std::vector<const char *>                 validation_layers;
	static const std::vector<VkValidationFeatureEnableEXT> validation_extensions;

  private:
	VkInstance m_handle = VK_NULL_HANDLE;

#ifdef _DEBUG
	bool m_debug_enable = true;
#else
	bool m_debug_enable = false;
#endif        // _DEBUG
};
}        // namespace Ilum