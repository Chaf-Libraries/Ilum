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

  private:
	VkInstance m_handle = VK_NULL_HANDLE;

#ifdef _DEBUG
	const std::vector<const char *>                 m_extensions            = {"VK_KHR_surface", "VK_KHR_win32_surface", "VK_EXT_debug_report", "VK_EXT_debug_utils"};
	const std::vector<const char *>           m_validation_layers     = {"VK_LAYER_KHRONOS_validation"};
	const std::vector<VkValidationFeatureEnableEXT> m_validation_extensions = {
	    VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT,
	    VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
	    VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,
	    VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT};

	bool m_debug_enable = true;
#else
	std::vector<const char *>                 m_extensions            = {"VK_KHR_surface", "VK_KHR_win32_surface"};
	std::vector<const char *>                 m_validation_layers     = {};
	std::vector<VkValidationFeatureEnableEXT> m_validation_extensions = {};

	bool m_debug_enable = false;
#endif        // _DEBUG
};
}        // namespace Ilum