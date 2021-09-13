#pragma once

#include <volk.h>

namespace Ilum::Vulkan
{
class FunctionEXT
{
  public:
	FunctionEXT() = default;

	~FunctionEXT() = default;

	static void initialzize(VkInstance instance);

	static PFN_vkCreateDebugUtilsMessengerEXT          create_debug_utils_messenger;
	static VkDebugUtilsMessengerEXT                    debug_utils_messenger;
	static PFN_vkDestroyDebugUtilsMessengerEXT         destroy_messenger;
	static PFN_vkSetDebugUtilsObjectTagEXT             set_debug_utils_object_tag;
	static PFN_vkSetDebugUtilsObjectNameEXT            set_debug_utils_object_name;
	static PFN_vkCmdBeginDebugUtilsLabelEXT            begin_debug_utils_label;
	static PFN_vkCmdEndDebugUtilsLabelEXT              end_debug_utils_label;
	static PFN_vkGetPhysicalDeviceMemoryProperties2KHR get_physical_device_memory_properties_2;
};
}        // namespace Ilum