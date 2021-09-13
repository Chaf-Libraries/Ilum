#include "FunctionEXT.hpp"

namespace Ilum::Vulkan
{
void FunctionEXT::initialzize(VkInstance instance)
{
	create_debug_utils_messenger            = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
	destroy_messenger                       = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
	set_debug_utils_object_tag              = reinterpret_cast<PFN_vkSetDebugUtilsObjectTagEXT>(vkGetInstanceProcAddr(instance, "vkSetDebugUtilsObjectTagEXT"));
	set_debug_utils_object_name             = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(vkGetInstanceProcAddr(instance, "vkSetDebugUtilsObjectNameEXT"));
	begin_debug_utils_label                 = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(vkGetInstanceProcAddr(instance, "vkCmdBeginDebugUtilsLabelEXT"));
	end_debug_utils_label                   = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(vkGetInstanceProcAddr(instance, "vkCmdEndDebugUtilsLabelEXT"));
	get_physical_device_memory_properties_2 = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2KHR>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceMemoryProperties2"));

	static PFN_vkCreateDebugUtilsMessengerEXT          create_debug_utils_messenger;
	static VkDebugUtilsMessengerEXT                    debug_utils_messenger;
	static PFN_vkDestroyDebugUtilsMessengerEXT         destroy_messenger;
	static PFN_vkSetDebugUtilsObjectTagEXT             set_debug_utils_object_tag;
	static PFN_vkSetDebugUtilsObjectNameEXT            set_debug_utils_object_name;
	static PFN_vkCmdBeginDebugUtilsLabelEXT            begin_debug_utils_label;
	static PFN_vkCmdEndDebugUtilsLabelEXT              end_debug_utils_label;
	static PFN_vkGetPhysicalDeviceMemoryProperties2KHR get_physical_device_memory_properties_2;
}
}        // namespace Ilum::Vulkan