#define VMA_IMPLEMENTATION
#define VOLK_IMPLEMENTATION

#include "Device.hpp"

#include <Core/Window.hpp>

namespace Ilum
{
PFN_vkCreateDebugUtilsMessengerEXT          RHIDevice::vkCreateDebugUtilsMessengerEXT          = nullptr;
VkDebugUtilsMessengerEXT                    RHIDevice::vkDebugUtilsMessengerEXT                = nullptr;
PFN_vkDestroyDebugUtilsMessengerEXT         RHIDevice::vkDestroyDebugUtilsMessengerEXT         = nullptr;
PFN_vkSetDebugUtilsObjectTagEXT             RHIDevice::vkSetDebugUtilsObjectTagEXT             = nullptr;
PFN_vkSetDebugUtilsObjectNameEXT            RHIDevice::vkSetDebugUtilsObjectNameEXT            = nullptr;
PFN_vkCmdBeginDebugUtilsLabelEXT            RHIDevice::vkCmdBeginDebugUtilsLabelEXT            = nullptr;
PFN_vkCmdEndDebugUtilsLabelEXT              RHIDevice::vkCmdEndDebugUtilsLabelEXT              = nullptr;
PFN_vkGetPhysicalDeviceMemoryProperties2KHR RHIDevice::vkGetPhysicalDeviceMemoryProperties2KHR = nullptr;

#ifdef _DEBUG
const std::vector<const char *>                 RHIDevice::s_instance_extensions   = {"VK_KHR_surface", "VK_KHR_win32_surface", "VK_EXT_debug_report", "VK_EXT_debug_utils"};
const std::vector<const char *>                 RHIDevice::s_validation_layers     = {"VK_LAYER_KHRONOS_validation"};
const std::vector<VkValidationFeatureEnableEXT> RHIDevice::s_validation_extensions = {
    VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT,
    VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
    VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT};
#else
const std::vector<const char *>                 RHIDevice::s_instance_extensions   = {"VK_KHR_surface", "VK_KHR_win32_surface"};
const std::vector<const char *>                 RHIDevice::s_validation_layers     = {};
const std::vector<VkValidationFeatureEnableEXT> RHIDevice::s_validation_extensions = {};
#endif        // _DEBUG

inline const std::vector<const char *> get_instance_extension_supported(const std::vector<const char *> &extensions)
{
	uint32_t extension_count = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);

	std::vector<VkExtensionProperties> device_extensions(extension_count);
	vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, device_extensions.data());

	std::vector<const char *> result;

	for (const auto &extension : extensions)
	{
		bool found = false;
		for (const auto &device_extension : device_extensions)
		{
			if (strcmp(extension, device_extension.extensionName) == 0)
			{
				result.emplace_back(extension);
				found = true;
				LOG_INFO("Enable instance extension: {}", extension);
				break;
			}
		}
		if (!found)
		{
			LOG_WARN("Instance extension {} is not supported", extension);
		}
	}

	return result;
}

inline bool check_layer_supported(const char *layer_name)
{
	uint32_t layer_count;
	vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

	std::vector<VkLayerProperties> layers(layer_count);
	vkEnumerateInstanceLayerProperties(&layer_count, layers.data());

	for (const auto &layer : layers)
	{
		if (strcmp(layer.layerName, layer_name) == 0)
		{
			return true;
		}
	}

	return false;
}

static inline VKAPI_ATTR VkBool32 VKAPI_CALL validation_callback(VkDebugUtilsMessageSeverityFlagBitsEXT msg_severity, VkDebugUtilsMessageTypeFlagsEXT msg_type, const VkDebugUtilsMessengerCallbackDataEXT *callback_data, void *user_data)
{
	if (msg_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
	{
		LOG_INFO(callback_data->pMessage);
	}
	else if (msg_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
	{
		LOG_WARN(callback_data->pMessage);
	}
	else if (msg_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
	{
		LOG_ERROR(callback_data->pMessage);
	}

	return VK_FALSE;
}

RHIDevice::RHIDevice(Window *window)
{
	CreateInstance();
}

RHIDevice::~RHIDevice()
{
	if (vkDestroyDebugUtilsMessengerEXT)
	{
		vkDestroyDebugUtilsMessengerEXT(m_instance, vkDebugUtilsMessengerEXT, nullptr);
	}
	vkDestroyInstance(m_instance, nullptr);
}

void RHIDevice::CreateInstance()
{
	// Initialize volk context
	volkInitialize();

	// Config application info
	VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};

	uint32_t sdk_version = VK_HEADER_VERSION_COMPLETE;
	uint32_t api_version = 0;

	PFN_vkEnumerateInstanceVersion enumerate_instance_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));

	if (enumerate_instance_version)
	{
		enumerate_instance_version(&api_version);
	}
	else
	{
		api_version = VK_VERSION_1_0;
	}

	if (sdk_version > api_version)
	{
		std::string sdk_version_str = std::to_string(VK_VERSION_MAJOR(sdk_version)) + "." + std::to_string(VK_VERSION_MINOR(sdk_version)) + "." + std::to_string(VK_VERSION_PATCH(sdk_version));
		std::string api_version_str = std::to_string(VK_VERSION_MAJOR(api_version)) + "." + std::to_string(VK_VERSION_MINOR(api_version)) + "." + std::to_string(VK_VERSION_PATCH(api_version));
		LOG_WARN("Using Vulkan {}, please upgrade your graphics driver to support Vulkan {}", api_version_str, sdk_version_str);
	}

	app_info.pApplicationName   = "IlumEngine";
	app_info.pEngineName        = "IlumEngine";
	app_info.engineVersion      = VK_MAKE_VERSION(0, 0, 1);
	app_info.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
	app_info.apiVersion         = std::min(sdk_version, api_version);

	// Check out extensions support
	std::vector<const char *> extensions_support = get_instance_extension_supported(s_instance_extensions);

	// Config instance info
	VkInstanceCreateInfo create_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	create_info.pApplicationInfo        = &app_info;
	create_info.enabledExtensionCount   = static_cast<uint32_t>(extensions_support.size());
	create_info.ppEnabledExtensionNames = extensions_support.data();
	create_info.enabledLayerCount       = 0;

	// Enable validation layers
#ifdef _DEBUG
	// Validation features
	VkValidationFeaturesEXT validation_features{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
	validation_features.enabledValidationFeatureCount = static_cast<uint32_t>(s_validation_extensions.size());
	validation_features.pEnabledValidationFeatures    = s_validation_extensions.data();

	// Enable validation layer
	for (auto &layer : s_validation_layers)
	{
		if (check_layer_supported(layer))
		{
			LOG_INFO("Enable validation layer: {}", layer);
			create_info.enabledLayerCount   = static_cast<uint32_t>(s_validation_layers.size());
			create_info.ppEnabledLayerNames = s_validation_layers.data();
			create_info.pNext               = &validation_features;
			break;
		}
		else
		{
			LOG_ERROR("Validation layer was required, but not avaliable, disabling debugging");
		}
	}
#endif        // _DEBUG

	// Create instance
	if (vkCreateInstance(&create_info, nullptr, &m_instance) != VK_SUCCESS)
	{
		LOG_ERROR("Failed to create vulkan instance!");
		return;
	}
	else
	{
		// Config to volk
		volkLoadInstance(m_instance);
	}

	// Initialize instance extension functions
	vkCreateDebugUtilsMessengerEXT          = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT"));
	vkDestroyDebugUtilsMessengerEXT         = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT"));
	vkSetDebugUtilsObjectTagEXT             = reinterpret_cast<PFN_vkSetDebugUtilsObjectTagEXT>(vkGetInstanceProcAddr(m_instance, "vkSetDebugUtilsObjectTagEXT"));
	vkSetDebugUtilsObjectNameEXT            = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(vkGetInstanceProcAddr(m_instance, "vkSetDebugUtilsObjectNameEXT"));
	vkCmdBeginDebugUtilsLabelEXT            = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(vkGetInstanceProcAddr(m_instance, "vkCmdBeginDebugUtilsLabelEXT"));
	vkCmdEndDebugUtilsLabelEXT              = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(vkGetInstanceProcAddr(m_instance, "vkCmdEndDebugUtilsLabelEXT"));
	vkGetPhysicalDeviceMemoryProperties2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2KHR>(vkGetInstanceProcAddr(m_instance, "vkGetPhysicalRHIDeviceMemoryProperties2KHR"));

	// Enable debugger
#ifdef _DEBUG
	if (vkCreateDebugUtilsMessengerEXT)
	{
		VkDebugUtilsMessengerCreateInfoEXT create_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
		create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		create_info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		create_info.pfnUserCallback = validation_callback;

		vkCreateDebugUtilsMessengerEXT(m_instance, &create_info, nullptr, &vkDebugUtilsMessengerEXT);
	}
#endif        // _DEBUG
}

void RHIDevice::CreatePhysicalDevice()
{
}

}        // namespace Ilum