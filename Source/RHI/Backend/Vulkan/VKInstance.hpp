#pragma once

#include "Vulkan.hpp"

#include <memory>
#include <vector>

namespace Ilum::RHI::Vulkan
{
class VKInstance
{
  public:
	VKInstance();

	~VKInstance();

	VKInstance(const VKInstance &) = delete;

	VKInstance &operator=(const VKInstance &) = delete;

	VKInstance(VKInstance &&) = delete;

	VKInstance &operator=(VKInstance &&) = delete;

	operator const VkInstance &() const;

	const VkInstance &GetHandle() const;

  public:
	// Extension functions
	static PFN_vkCreateDebugUtilsMessengerEXT          CreateDebugUtilsMessengerEXT;
	static VkDebugUtilsMessengerEXT                    DebugUtilsMessengerEXT;
	static PFN_vkDestroyDebugUtilsMessengerEXT         DestroyDebugUtilsMessengerEXT;
	static PFN_vkSetDebugUtilsObjectTagEXT             SetDebugUtilsObjectTagEXT;
	static PFN_vkSetDebugUtilsObjectNameEXT            SetDebugUtilsObjectNameEXT;
	static PFN_vkCmdBeginDebugUtilsLabelEXT            CmdBeginDebugUtilsLabelEXT;
	static PFN_vkCmdEndDebugUtilsLabelEXT              CmdEndDebugUtilsLabelEXT;
	static PFN_vkGetPhysicalDeviceMemoryProperties2KHR GetPhysicalDeviceMemoryProperties2KHR;

  public:
	// Extensions
	static const std::vector<const char *>                 s_extensions;
	static const std::vector<const char *>                 s_validation_layers;
	static const std::vector<VkValidationFeatureEnableEXT> s_validation_extensions;

  private:
	VkInstance m_handle = VK_NULL_HANDLE;

#ifdef _DEBUG
	bool m_debug_enable = true;
#else
	bool m_debug_enable = false;
#endif        // _DEBUG
};
}        // namespace Ilum::RHI::Vulkan