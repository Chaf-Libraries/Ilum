#include "Device.hpp"

namespace Ilum::Vulkan
{
// Extension Function
static PFN_vkCreateDebugUtilsMessengerEXT  vkCreateDebugUtilsMessengerEXT;
static VkDebugUtilsMessengerEXT            vkDebugUtilsMessengerEXT;
static PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT;
PFN_vkSetDebugUtilsObjectTagEXT            vkSetDebugUtilsObjectTagEXT;
PFN_vkSetDebugUtilsObjectNameEXT           vkSetDebugUtilsObjectNameEXT;
PFN_vkCmdBeginDebugUtilsLabelEXT           vkCmdBeginDebugUtilsLabelEXT;
PFN_vkCmdEndDebugUtilsLabelEXT             vkCmdEndDebugUtilsLabelEXT;
PFN_vkGetSemaphoreWin32HandleKHR           vkGetSemaphoreWin32HandleKHR;
PFN_vkGetMemoryWin32HandleKHR              vkGetMemoryWin32HandleKHR;

// Vulkan Extension
static const std::vector<const char *> InstanceExtensions =
#ifdef DEBUG
    {"VK_KHR_surface", "VK_KHR_win32_surface", "VK_EXT_debug_report", "VK_EXT_debug_utils"};
#else
    {"VK_KHR_surface", "VK_KHR_win32_surface"};
#endif

static const std::vector<const char *> ValidationLayers =
#ifdef DEBUG
    {"VK_LAYER_KHRONOS_validation"};
#else
    {};
#endif        // DEBUG

const std::vector<VkValidationFeatureEnableEXT> ValidationExtensions =
#ifdef DEBUG
    {VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT};
#else
    {};
#endif        // DEBUG

static const std::vector<const char *> DeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    VK_EXT_MESH_SHADER_EXTENSION_NAME,
    VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME,
    VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
    VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
#ifdef _WIN64
    VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME
#else
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
};

// Utilities Function
inline const std::vector<const char *> GetInstanceExtensionSupported(const std::vector<const char *> &extensions)
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
				break;
			}
		}
	}

	return result;
}

inline bool CheckLayerSupported(const char *layer_name)
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

inline uint32_t ScorePhysicalDevice(VkPhysicalDevice physical_device, const std::vector<const char *> &device_extensions, std::vector<const char *> &support_device_extensions)
{
	uint32_t score = 0;

	// Check extensions
	uint32_t device_extension_properties_count = 0;
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, nullptr);

	std::vector<VkExtensionProperties> extension_properties(device_extension_properties_count);
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, extension_properties.data());

	for (auto &device_extension : device_extensions)
	{
		for (auto &support_extension : extension_properties)
		{
			if (std::strcmp(device_extension, support_extension.extensionName) == 0)
			{
				support_device_extensions.push_back(device_extension);
				score += 100;
				break;
			}
		}
	}

	VkPhysicalDeviceProperties properties = {};

	vkGetPhysicalDeviceProperties(physical_device, &properties);

	// Score discrete gpu
	if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
	{
		score += 1000;
	}

	score += properties.limits.maxImageDimension2D;
	return score;
}

inline VkPhysicalDevice SelectPhysicalDevice(const std::vector<VkPhysicalDevice> &physical_devices, const std::vector<const char *> &device_extensions, std::vector<const char *> &support_device_extensions)
{
	// Score - GPU
	uint32_t         score  = 0;
	VkPhysicalDevice handle = VK_NULL_HANDLE;
	for (auto &gpu : physical_devices)
	{
		std::vector<const char *> support_extensions;

		uint32_t tmp_score = ScorePhysicalDevice(gpu, device_extensions, support_extensions);
		if (tmp_score > score)
		{
			score  = tmp_score;
			handle = gpu;

			support_device_extensions = std::move(support_extensions);
		}
	}

	return handle;
}

inline std::optional<uint32_t> GetQueueFamilyIndex(const std::vector<VkQueueFamilyProperties> &queue_family_properties, VkQueueFlagBits queue_flag)
{
	// Dedicated queue for compute
	// Try to find a queue family index that supports compute but not graphics
	if (queue_flag & VK_QUEUE_COMPUTE_BIT)
	{
		for (uint32_t i = 0; i < static_cast<uint32_t>(queue_family_properties.size()); i++)
		{
			if ((queue_family_properties[i].queueFlags & queue_flag) && ((queue_family_properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0))
			{
				return i;
				break;
			}
		}
	}

	// Dedicated queue for transfer
	// Try to find a queue family index that supports transfer but not graphics and compute
	if (queue_flag & VK_QUEUE_TRANSFER_BIT)
	{
		for (uint32_t i = 0; i < static_cast<uint32_t>(queue_family_properties.size()); i++)
		{
			if ((queue_family_properties[i].queueFlags & queue_flag) && ((queue_family_properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) && ((queue_family_properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0))
			{
				return i;
				break;
			}
		}
	}

	// For other queue types or if no separate compute queue is present, return the first one to support the requested flags
	for (uint32_t i = 0; i < static_cast<uint32_t>(queue_family_properties.size()); i++)
	{
		if (queue_family_properties[i].queueFlags & queue_flag)
		{
			return i;
			break;
		}
	}

	return std::optional<uint32_t>();
}

inline const std::vector<const char *> GetDeviceExtensionSupport(VkPhysicalDevice physical_device, const std::vector<const char *> &extensions)
{
	std::vector<const char *> result;

	uint32_t device_extension_properties_count = 0;
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, nullptr);

	std::vector<VkExtensionProperties> extension_properties(device_extension_properties_count);
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, extension_properties.data());

	for (auto &device_extension : extensions)
	{
		bool enable = false;
		for (auto &support_extension : extension_properties)
		{
			if (std::strcmp(device_extension, support_extension.extensionName) == 0)
			{
				result.push_back(device_extension);
				enable = true;
				break;
			}
		}
	}

	return result;
}

static inline VKAPI_ATTR VkBool32 VKAPI_CALL ValidationCallback(VkDebugUtilsMessageSeverityFlagBitsEXT msg_severity, VkDebugUtilsMessageTypeFlagsEXT msg_type, const VkDebugUtilsMessengerCallbackDataEXT *callback_data, void *user_data)
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

void Device::CreateInstance()
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
		LOG_WARN("Driver support version {} is higher than API version {}, upgrade your VulkanSDK!", sdk_version_str, api_version_str);
	}

	app_info.pApplicationName   = "IlumEngine";
	app_info.pEngineName        = "IlumEngine";
	app_info.engineVersion      = VK_MAKE_VERSION(0, 0, 1);
	app_info.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
	app_info.apiVersion         = std::max(sdk_version, api_version);

	// Check out extensions support
	m_supported_instance_extensions = GetInstanceExtensionSupported(InstanceExtensions);

	// Config instance info
	VkInstanceCreateInfo create_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	create_info.pApplicationInfo        = &app_info;
	create_info.enabledExtensionCount   = static_cast<uint32_t>(m_supported_instance_extensions.size());
	create_info.ppEnabledExtensionNames = m_supported_instance_extensions.data();
	create_info.enabledLayerCount       = 0;

	// Enable validation layers
#ifdef DEBUG
	VkValidationFeaturesEXT validation_features{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
	validation_features.enabledValidationFeatureCount = static_cast<uint32_t>(ValidationExtensions.size());
	validation_features.pEnabledValidationFeatures    = ValidationExtensions.data();

	// Enable validation layer
	for (auto &layer : ValidationLayers)
	{
		if (CheckLayerSupported(layer))
		{
			create_info.enabledLayerCount   = static_cast<uint32_t>(ValidationLayers.size());
			create_info.ppEnabledLayerNames = ValidationLayers.data();
			create_info.pNext               = &validation_features;
			break;
		}
		else
		{
			LOG_ERROR("Validation layer was required, but not avaliable, disabling debugging");
		}
	}
#endif        // DEBUG

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
	vkCreateDebugUtilsMessengerEXT  = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT"));
	vkDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT"));
	vkSetDebugUtilsObjectTagEXT     = reinterpret_cast<PFN_vkSetDebugUtilsObjectTagEXT>(vkGetInstanceProcAddr(m_instance, "vkSetDebugUtilsObjectTagEXT"));
	vkSetDebugUtilsObjectNameEXT    = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(vkGetInstanceProcAddr(m_instance, "vkSetDebugUtilsObjectNameEXT"));
	vkCmdBeginDebugUtilsLabelEXT    = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(vkGetInstanceProcAddr(m_instance, "vkCmdBeginDebugUtilsLabelEXT"));
	vkCmdEndDebugUtilsLabelEXT      = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(vkGetInstanceProcAddr(m_instance, "vkCmdEndDebugUtilsLabelEXT"));
	vkGetSemaphoreWin32HandleKHR    = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(vkGetInstanceProcAddr(m_instance, "vkGetSemaphoreWin32HandleKHR"));
	vkGetMemoryWin32HandleKHR       = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetInstanceProcAddr(m_instance, "vkGetMemoryWin32HandleKHR"));

	// Enable debugger
#ifdef DEBUG
	if (vkCreateDebugUtilsMessengerEXT)
	{
		VkDebugUtilsMessengerCreateInfoEXT create_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
		create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		create_info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		create_info.pfnUserCallback = ValidationCallback;

		vkCreateDebugUtilsMessengerEXT(m_instance, &create_info, nullptr, &vkDebugUtilsMessengerEXT);
	}
#endif        // DEBUG
}

void Device::CreatePhysicalDevice()
{
	uint32_t physical_device_count = 0;
	vkEnumeratePhysicalDevices(m_instance, &physical_device_count, nullptr);

	// Get all physical devices
	std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
	vkEnumeratePhysicalDevices(m_instance, &physical_device_count, physical_devices.data());

	// Select suitable physical device
	m_physical_device = SelectPhysicalDevice(physical_devices, DeviceExtensions, m_supported_device_extensions);

	VkPhysicalDeviceProperties properties = {};
	vkGetPhysicalDeviceProperties(m_physical_device, &properties);
	m_name = properties.deviceName;
}

void Device::CreateLogicalDevice()
{
	// Queue supporting
	uint32_t queue_family_property_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &queue_family_property_count, nullptr);
	std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_property_count);
	vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &queue_family_property_count, queue_family_properties.data());

	std::optional<uint32_t> graphics_family, compute_family, transfer_family, present_family;

	graphics_family = GetQueueFamilyIndex(queue_family_properties, VK_QUEUE_GRAPHICS_BIT);
	transfer_family = GetQueueFamilyIndex(queue_family_properties, VK_QUEUE_TRANSFER_BIT);
	compute_family  = GetQueueFamilyIndex(queue_family_properties, VK_QUEUE_COMPUTE_BIT);

	VkQueueFlags support_queues = 0;

	if (graphics_family.has_value())
	{
		m_graphics_family = graphics_family.value();
		support_queues |= VK_QUEUE_GRAPHICS_BIT;
	}

	if (compute_family.has_value())
	{
		m_compute_family = compute_family.value();
		support_queues |= VK_QUEUE_COMPUTE_BIT;
	}

	if (transfer_family.has_value())
	{
		m_transfer_family = transfer_family.value();
		support_queues |= VK_QUEUE_TRANSFER_BIT;
	}

	if (!graphics_family)
	{
		throw std::runtime_error("Failed to find queue graphics family support!");
	}

	// Create device queue
	std::vector<VkDeviceQueueCreateInfo> queue_create_infos;

	uint32_t max_count = 0;
	for (auto &queue_family_property : queue_family_properties)
	{
		max_count = max_count < queue_family_property.queueCount ? queue_family_property.queueCount : max_count;
	}

	std::vector<float> queue_priorities(max_count, 1.f);

	if (support_queues & VK_QUEUE_GRAPHICS_BIT)
	{
		VkDeviceQueueCreateInfo graphics_queue_create_info = {};
		graphics_queue_create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		graphics_queue_create_info.queueFamilyIndex        = m_graphics_family;
		graphics_queue_create_info.queueCount              = queue_family_properties[m_graphics_family].queueCount;
		graphics_queue_create_info.pQueuePriorities        = queue_priorities.data();
		queue_create_infos.emplace_back(graphics_queue_create_info);
		m_graphics_queue_count = queue_family_properties[m_graphics_family].queueCount;
	}
	else
	{
		m_graphics_family = 0;
	}

	if (support_queues & VK_QUEUE_COMPUTE_BIT && m_compute_family != m_graphics_family)
	{
		VkDeviceQueueCreateInfo compute_queue_create_info = {};
		compute_queue_create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		compute_queue_create_info.queueFamilyIndex        = m_compute_family;
		compute_queue_create_info.queueCount              = queue_family_properties[m_compute_family].queueCount;
		compute_queue_create_info.pQueuePriorities        = queue_priorities.data();
		queue_create_infos.emplace_back(compute_queue_create_info);
		m_compute_queue_count = queue_family_properties[m_compute_family].queueCount;
	}
	else
	{
		m_compute_family      = m_graphics_family;
		m_compute_queue_count = m_graphics_queue_count;
	}

	if (support_queues & VK_QUEUE_TRANSFER_BIT && m_transfer_family != m_graphics_family && m_transfer_family != m_compute_family)
	{
		VkDeviceQueueCreateInfo transfer_queue_create_info = {};
		transfer_queue_create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		transfer_queue_create_info.queueFamilyIndex        = m_transfer_family;
		transfer_queue_create_info.queueCount              = queue_family_properties[m_transfer_family].queueCount;
		transfer_queue_create_info.pQueuePriorities        = queue_priorities.data();
		queue_create_infos.emplace_back(transfer_queue_create_info);
		m_transfer_queue_count = queue_family_properties[m_transfer_family].queueCount;
	}
	else
	{
		m_transfer_family      = m_graphics_family;
		m_transfer_queue_count = m_graphics_queue_count;
	}

	// Enable logical device features
	VkPhysicalDeviceFeatures2        physical_device_features          = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	VkPhysicalDeviceVulkan12Features physical_device_vulkan12_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	VkPhysicalDeviceVulkan13Features physical_device_vulkan13_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};

	physical_device_features.pNext          = &physical_device_vulkan12_features;
	physical_device_vulkan12_features.pNext = &physical_device_vulkan13_features;

	vkGetPhysicalDeviceFeatures2(m_physical_device, &physical_device_features);

	VkPhysicalDeviceFeatures2        physical_device_features_enable          = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	VkPhysicalDeviceVulkan12Features physical_device_vulkan12_features_enable = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	VkPhysicalDeviceVulkan13Features physical_device_vulkan13_features_enable = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};

#define ENABLE_DEVICE_FEATURE(device_feature, device_feature_enable, feature) \
	if (device_feature.feature)                                               \
	{                                                                         \
		device_feature_enable.feature = VK_TRUE;                              \
		m_supported_device_features.push_back(#feature);                      \
	}                                                                         \
	else                                                                      \
	{                                                                         \
		LOG_WARN("Device feature {} is not supported", #feature);             \
	}

	ENABLE_DEVICE_FEATURE(physical_device_features.features, physical_device_features_enable.features, sampleRateShading);
	ENABLE_DEVICE_FEATURE(physical_device_features.features, physical_device_features_enable.features, fillModeNonSolid);
	ENABLE_DEVICE_FEATURE(physical_device_features.features, physical_device_features_enable.features, wideLines);
	ENABLE_DEVICE_FEATURE(physical_device_features.features, physical_device_features_enable.features, samplerAnisotropy);
	ENABLE_DEVICE_FEATURE(physical_device_features.features, physical_device_features_enable.features, vertexPipelineStoresAndAtomics);
	ENABLE_DEVICE_FEATURE(physical_device_features.features, physical_device_features_enable.features, fragmentStoresAndAtomics);
	ENABLE_DEVICE_FEATURE(physical_device_features.features, physical_device_features_enable.features, shaderStorageImageExtendedFormats);
	ENABLE_DEVICE_FEATURE(physical_device_features.features, physical_device_features_enable.features, shaderStorageImageWriteWithoutFormat);
	ENABLE_DEVICE_FEATURE(physical_device_features.features, physical_device_features_enable.features, geometryShader);
	ENABLE_DEVICE_FEATURE(physical_device_features.features, physical_device_features_enable.features, tessellationShader);
	ENABLE_DEVICE_FEATURE(physical_device_features.features, physical_device_features_enable.features, multiViewport);
	ENABLE_DEVICE_FEATURE(physical_device_features.features, physical_device_features_enable.features, imageCubeArray);
	ENABLE_DEVICE_FEATURE(physical_device_features.features, physical_device_features_enable.features, robustBufferAccess);
	ENABLE_DEVICE_FEATURE(physical_device_features.features, physical_device_features_enable.features, multiDrawIndirect);
	ENABLE_DEVICE_FEATURE(physical_device_features.features, physical_device_features_enable.features, drawIndirectFirstInstance);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, drawIndirectCount);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, descriptorIndexing);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, shaderInputAttachmentArrayDynamicIndexing);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, shaderUniformTexelBufferArrayDynamicIndexing);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, shaderStorageTexelBufferArrayDynamicIndexing);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, shaderUniformBufferArrayNonUniformIndexing);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, shaderSampledImageArrayNonUniformIndexing);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, shaderStorageBufferArrayNonUniformIndexing);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, shaderStorageImageArrayNonUniformIndexing);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, shaderInputAttachmentArrayNonUniformIndexing);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, shaderUniformTexelBufferArrayNonUniformIndexing);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, shaderStorageTexelBufferArrayNonUniformIndexing);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, descriptorBindingUniformBufferUpdateAfterBind);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, descriptorBindingSampledImageUpdateAfterBind);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, descriptorBindingStorageImageUpdateAfterBind);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, descriptorBindingStorageBufferUpdateAfterBind);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, descriptorBindingUniformTexelBufferUpdateAfterBind);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, descriptorBindingStorageTexelBufferUpdateAfterBind);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, descriptorBindingPartiallyBound);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, runtimeDescriptorArray);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, timelineSemaphore);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, bufferDeviceAddress);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, shaderOutputViewportIndex);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan12_features, physical_device_vulkan12_features_enable, shaderOutputLayer);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan13_features, physical_device_vulkan13_features_enable, dynamicRendering);
	ENABLE_DEVICE_FEATURE(physical_device_vulkan13_features, physical_device_vulkan13_features_enable, maintenance4);

	m_vulkan_feature_support[VulkanFeature::DynamicRendering] = physical_device_vulkan13_features_enable.dynamicRendering;

	// Get support extensions
	auto support_extensions = GetDeviceExtensionSupport(m_physical_device, DeviceExtensions);

	{
		m_feature_support[RHIFeature::RayTracing]       = true;
		std::vector<const char *> raytracing_extensions = {
		    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
		    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
		    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
		    VK_KHR_RAY_QUERY_EXTENSION_NAME};

		for (auto &raytracing_extension : raytracing_extensions)
		{
			bool found = false;
			for (auto &extension : m_supported_device_extensions)
			{
				if (strcmp(raytracing_extension, extension) == 0)
				{
					found = true;
					continue;
				}
			}
			if (!found)
			{
				m_feature_support[RHIFeature::RayTracing] = false;
				break;
			}
		}
	};
	{
		m_feature_support[RHIFeature::MeshShading] = false;
		for (auto &extension : m_supported_device_extensions)
		{
			if (strcmp(VK_EXT_MESH_SHADER_EXTENSION_NAME, extension) == 0)
			{
				m_feature_support[RHIFeature::MeshShading] = true;
				break;
			}
		}
	}
	{
		m_feature_support[RHIFeature::BufferDeviceAddress] = false;
		for (auto &feature : m_supported_device_features)
		{
			if (strcmp("bufferDeviceAddress", feature) == 0)
			{
				m_feature_support[RHIFeature::BufferDeviceAddress] = true;
				break;
			}
		}
	}
	{
		m_feature_support[RHIFeature::Bindless] = false;
		for (auto &extension : m_supported_device_features)
		{
			if (strcmp("descriptorIndexing", extension) == 0)
			{
				m_feature_support[RHIFeature::Bindless] = true;
				break;
			}
		}
	}

	VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_feature = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
	VkPhysicalDeviceRayTracingPipelineFeaturesKHR    ray_tracing_pipeline_feature   = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
	VkPhysicalDeviceRayQueryFeaturesKHR              ray_query_features             = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
	VkPhysicalDeviceMeshShaderFeaturesEXT            mesh_shader_feature            = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};

	void  *feature_ptr_head = nullptr;
	void **feature_ptr_tail = nullptr;

	if (IsFeatureSupport(RHIFeature::RayTracing))
	{
		acceleration_structure_feature.accelerationStructure = VK_TRUE;
		ray_tracing_pipeline_feature.rayTracingPipeline      = VK_TRUE;
		ray_query_features.rayQuery                          = VK_TRUE;

		acceleration_structure_feature.pNext = &ray_tracing_pipeline_feature;
		ray_tracing_pipeline_feature.pNext   = &ray_query_features;

		feature_ptr_head = &acceleration_structure_feature;
		feature_ptr_tail = &ray_query_features.pNext;
	}

	if (IsFeatureSupport(RHIFeature::MeshShading))
	{
		mesh_shader_feature.meshShader          = VK_TRUE;
		mesh_shader_feature.taskShader          = VK_TRUE;
		mesh_shader_feature.multiviewMeshShader = VK_TRUE;

		if (!feature_ptr_head)
		{
			feature_ptr_head = &mesh_shader_feature;
		}
		else
		{
			*feature_ptr_tail = &mesh_shader_feature;
		}
		feature_ptr_tail = &mesh_shader_feature.pNext;
	}

	physical_device_vulkan12_features_enable.pNext = &physical_device_vulkan13_features_enable;
	physical_device_vulkan13_features_enable.pNext = feature_ptr_head;

	// Create device
	VkDeviceCreateInfo device_create_info = {};

	device_create_info.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	device_create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
	device_create_info.pQueueCreateInfos    = queue_create_infos.data();
	if (!ValidationLayers.empty())
	{
		device_create_info.enabledLayerCount   = static_cast<uint32_t>(ValidationLayers.size());
		device_create_info.ppEnabledLayerNames = ValidationLayers.data();
	}
	device_create_info.enabledExtensionCount   = static_cast<uint32_t>(support_extensions.size());
	device_create_info.ppEnabledExtensionNames = support_extensions.data();
	device_create_info.pEnabledFeatures        = &physical_device_features.features;
	device_create_info.pNext                   = &physical_device_vulkan12_features_enable;

	if (vkCreateDevice(m_physical_device, &device_create_info, nullptr, &m_logical_device) != VK_SUCCESS)
	{
		LOG_ERROR("Failed to create logical device!");
		return;
	}

	// Volk load context
	volkLoadDevice(m_logical_device);

	// Create Vma allocator
	VmaVulkanFunctions vma_vulkan_func{};
	vma_vulkan_func.vkGetInstanceProcAddr               = vkGetInstanceProcAddr;
	vma_vulkan_func.vkGetDeviceProcAddr                 = vkGetDeviceProcAddr;
	vma_vulkan_func.vkAllocateMemory                    = vkAllocateMemory;
	vma_vulkan_func.vkBindBufferMemory                  = vkBindBufferMemory;
	vma_vulkan_func.vkBindImageMemory                   = vkBindImageMemory;
	vma_vulkan_func.vkCreateBuffer                      = vkCreateBuffer;
	vma_vulkan_func.vkCreateImage                       = vkCreateImage;
	vma_vulkan_func.vkDestroyBuffer                     = vkDestroyBuffer;
	vma_vulkan_func.vkDestroyImage                      = vkDestroyImage;
	vma_vulkan_func.vkFlushMappedMemoryRanges           = vkFlushMappedMemoryRanges;
	vma_vulkan_func.vkFreeMemory                        = vkFreeMemory;
	vma_vulkan_func.vkGetBufferMemoryRequirements       = vkGetBufferMemoryRequirements;
	vma_vulkan_func.vkGetImageMemoryRequirements        = vkGetImageMemoryRequirements;
	vma_vulkan_func.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
	vma_vulkan_func.vkGetPhysicalDeviceProperties       = vkGetPhysicalDeviceProperties;
	vma_vulkan_func.vkInvalidateMappedMemoryRanges      = vkInvalidateMappedMemoryRanges;
	vma_vulkan_func.vkMapMemory                         = vkMapMemory;
	vma_vulkan_func.vkUnmapMemory                       = vkUnmapMemory;
	vma_vulkan_func.vkCmdCopyBuffer                     = vkCmdCopyBuffer;

	VmaAllocatorCreateInfo allocator_info = {};
	allocator_info.physicalDevice         = m_physical_device;
	allocator_info.device                 = m_logical_device;
	allocator_info.instance               = m_instance;
	allocator_info.vulkanApiVersion       = VK_API_VERSION_1_3;
	allocator_info.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

	VkPhysicalDeviceMemoryProperties memory_properties = {};
	vkGetPhysicalDeviceMemoryProperties(m_physical_device, &memory_properties);

	allocator_info.pVulkanFunctions = &vma_vulkan_func;

	if (vmaCreateAllocator(&allocator_info, &m_allocator) != VK_SUCCESS)
	{
		LOG_FATAL("Failed to create vulkan memory allocator");
	}
}

Device::Device() :
    RHIDevice("Vulkan")
{
	LOG_INFO("Initializing RHI backend Vulkan...");

	CreateInstance();
	CreatePhysicalDevice();
	CreateLogicalDevice();
}

Device::~Device()
{
	vkDeviceWaitIdle(m_logical_device);

	if (m_allocator)
	{
		vmaDestroyAllocator(m_allocator);
	}

	if (m_logical_device)
	{
		vkDestroyDevice(m_logical_device, nullptr);
	}

	if (vkDebugUtilsMessengerEXT)
	{
		vkDestroyDebugUtilsMessengerEXT(m_instance, vkDebugUtilsMessengerEXT, nullptr);
	}

	if (m_instance)
	{
		vkDestroyInstance(m_instance, nullptr);
	}
}

void Device::WaitIdle()
{
	vkDeviceWaitIdle(m_logical_device);
}

bool Device::IsFeatureSupport(RHIFeature feature)
{
	return m_feature_support[feature];
}

bool Device::IsFeatureSupport(VulkanFeature feature)
{
	return m_vulkan_feature_support[feature];
}

VkInstance Device::GetInstance() const
{
	return m_instance;
}

VkPhysicalDevice Device::GetPhysicalDevice() const
{
	return m_physical_device;
}

VkDevice Device::GetDevice() const
{
	return m_logical_device;
}

VmaAllocator Device::GetAllocator() const
{
	return m_allocator;
}

uint32_t Device::GetQueueFamily(RHIQueueFamily family)
{
	switch (family)
	{
		case Ilum::RHIQueueFamily::Graphics:
			return m_graphics_family;
		case Ilum::RHIQueueFamily::Compute:
			return m_compute_family;
		case Ilum::RHIQueueFamily::Transfer:
			return m_transfer_family;
		default:
			break;
	}
	return m_graphics_family;
}

uint32_t Device::GetQueueCount(RHIQueueFamily family)
{
	switch (family)
	{
		case Ilum::RHIQueueFamily::Graphics:
			return m_graphics_queue_count;
		case Ilum::RHIQueueFamily::Compute:
			return m_compute_queue_count;
		case Ilum::RHIQueueFamily::Transfer:
			return m_transfer_queue_count;
		default:
			break;
	}
	return m_graphics_queue_count;
}

void Device::SetVulkanObjectName(const VkDebugUtilsObjectNameInfoEXT &info)
{
#ifdef DEBUG
	vkSetDebugUtilsObjectNameEXT(m_logical_device, &info);
#endif        // DEBUG
}

void Device::BeginDebugUtilsLabel(VkCommandBuffer cmd_buffer, const VkDebugUtilsLabelEXT &label)
{
#ifdef DEBUG
	vkCmdBeginDebugUtilsLabelEXT(cmd_buffer, &label);
#endif        // DEBUG
}

void Device::EndDebugUtilsLabel(VkCommandBuffer cmd_buffer)
{
#ifdef DEBUG
	vkCmdEndDebugUtilsLabelEXT(cmd_buffer);
#endif        // DEBUG
}

void Device::GetSemaphoreWin32Handle(const VkSemaphoreGetWin32HandleInfoKHR *handle_info, HANDLE *handle)
{
	vkGetSemaphoreWin32HandleKHR(m_logical_device, handle_info, handle);
}

void Device::GetMemoryWin32Handle(const VkMemoryGetWin32HandleInfoKHR *handle_info, HANDLE *handle)
{
	vkGetMemoryWin32HandleKHR(m_logical_device, handle_info, handle);
}
}        // namespace Ilum::Vulkan