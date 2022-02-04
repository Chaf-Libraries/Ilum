#include "Device.hpp"
#include "Command.hpp"
#include "RenderContext.hpp"

#include <Core/Hash.hpp>
#include <Core/Window.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
#include <map>
#include <optional>
#include <sstream>

namespace Ilum::Vulkan
{
// Instance extension function
PFN_vkCreateDebugUtilsMessengerEXT          Instance::CreateDebugUtilsMessengerEXT          = nullptr;
VkDebugUtilsMessengerEXT                    Instance::DebugUtilsMessengerEXT                = nullptr;
PFN_vkDestroyDebugUtilsMessengerEXT         Instance::DestroyDebugUtilsMessengerEXT         = nullptr;
PFN_vkSetDebugUtilsObjectTagEXT             Instance::SetDebugUtilsObjectTagEXT             = nullptr;
PFN_vkSetDebugUtilsObjectNameEXT            Instance::SetDebugUtilsObjectNameEXT            = nullptr;
PFN_vkCmdBeginDebugUtilsLabelEXT            Instance::CmdBeginDebugUtilsLabelEXT            = nullptr;
PFN_vkCmdEndDebugUtilsLabelEXT              Instance::CmdEndDebugUtilsLabelEXT              = nullptr;
PFN_vkGetPhysicalDeviceMemoryProperties2KHR Instance::GetPhysicalDeviceMemoryProperties2KHR = nullptr;

// Instance extensions
#ifdef _DEBUG
const std::vector<const char *>                 Instance::s_extensions            = {"VK_KHR_surface", "VK_KHR_win32_surface", "VK_EXT_debug_report", "VK_EXT_debug_utils"};
const std::vector<const char *>                 Instance::s_validation_layers     = {"VK_LAYER_KHRONOS_validation"};
const std::vector<VkValidationFeatureEnableEXT> Instance::s_validation_extensions = {
    VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT,
    VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
    VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT};
#else
const std::vector<const char *>                 Instance::s_extensions            = {"VK_KHR_surface", "VK_KHR_win32_surface"};
const std::vector<const char *>                 Instance::s_validation_layers     = {};
const std::vector<VkValidationFeatureEnableEXT> Instance::s_validation_extensions = {};
#endif        // _DEBUG

// Device extensions
const std::vector<const char *> Device::s_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME};

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

inline uint32_t ScorePhysicalDevice(VkPhysicalDevice physical_device)
{
	uint32_t score = 0;

	// Check extensions
	uint32_t device_extension_properties_count = 0;
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, nullptr);

	std::vector<VkExtensionProperties> extension_properties(device_extension_properties_count);
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, extension_properties.data());

	for (auto &device_extension : Device::s_extensions)
	{
		for (auto &support_extension : extension_properties)
		{
			if (std::strcmp(device_extension, support_extension.extensionName) == 0)
			{
				score += 100;
				break;
			}
		}
	}

	VkPhysicalDeviceProperties       properties        = {};
	VkPhysicalDeviceFeatures         features          = {};
	VkPhysicalDeviceMemoryProperties memory_properties = {};

	vkGetPhysicalDeviceProperties(physical_device, &properties);
	vkGetPhysicalDeviceFeatures(physical_device, &features);
	vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

	// Logging gpu
	std::stringstream ss;
	ss << "\nFound physical device [" << properties.deviceID << "]\n";
	ss << "Name: " << properties.deviceName << "\n";
	ss << "Type: ";
	switch (static_cast<int32_t>(properties.deviceType))
	{
		case 1:
			ss << "Integrated\n";
			break;
		case 2:
			ss << "Discrete\n";
			break;
		case 3:
			ss << "Virtual\n";
			break;
		case 4:
			ss << "CPU\n";
			break;
		default:
			ss << "Other " << properties.deviceType << "\n";
	}

	ss << "Vendor: ";
	switch (properties.vendorID)
	{
		case 0x8086:
			ss << "Intel\n";
			break;
		case 0x10DE:
			ss << "Nvidia\n";
			break;
		case 0x1002:
			ss << "AMD\n";
			break;
		default:
			ss << properties.vendorID << "\n";
	}

	uint32_t supportedVersion[3] = {
	    VK_VERSION_MAJOR(properties.apiVersion),
	    VK_VERSION_MINOR(properties.apiVersion),
	    VK_VERSION_PATCH(properties.apiVersion)};

	ss << "API Version: " << supportedVersion[0] << "." << supportedVersion[1] << "." << supportedVersion[2] << '\n';

	// Score discrete gpu
	if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
	{
		score += 1000;
	}

	score += properties.limits.maxImageDimension2D;
	return score;
}

inline VkPhysicalDevice SelectPhysicalDevice(const std::vector<VkPhysicalDevice> &physical_devices)
{
	// Score - GPU
	std::map<uint32_t, VkPhysicalDevice, std::greater<uint32_t>> scores;
	for (auto &gpu : physical_devices)
	{
		scores.emplace(ScorePhysicalDevice(gpu), gpu);
	}

	if (scores.empty())
	{
		return VK_NULL_HANDLE;
	}

	return scores.begin()->second;
}

inline VkSampleCountFlagBits GetMaxSampleCount(VkPhysicalDevice physical_device)
{
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(physical_device, &properties);

	auto counts = std::min(properties.limits.framebufferColorSampleCounts, properties.limits.framebufferDepthSampleCounts);

	const std::vector<VkSampleCountFlagBits> sample_count_flag = {
	    VK_SAMPLE_COUNT_64_BIT,
	    VK_SAMPLE_COUNT_32_BIT,
	    VK_SAMPLE_COUNT_16_BIT,
	    VK_SAMPLE_COUNT_8_BIT,
	    VK_SAMPLE_COUNT_4_BIT,
	    VK_SAMPLE_COUNT_2_BIT,
	    VK_SAMPLE_COUNT_1_BIT};

	for (const auto &flag : sample_count_flag)
	{
		if (counts & flag)
		{
			return flag;
		}
	}

	return VK_SAMPLE_COUNT_1_BIT;
}

inline const std::vector<const char *> GetDeviceExtensionSupport(const PhysicalDevice &physical_device, const std::vector<const char *> &extensions)
{
	std::vector<const char *> result;

	uint32_t device_extension_properties_count = 0;
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, nullptr);

	std::vector<VkExtensionProperties> extension_properties(device_extension_properties_count);
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, extension_properties.data());

	for (auto &device_extension : Device::s_extensions)
	{
		bool enable = false;
		for (auto &support_extension : extension_properties)
		{
			if (std::strcmp(device_extension, support_extension.extensionName) == 0)
			{
				result.push_back(device_extension);
				enable = true;
				LOG_INFO("Enable device extension: {}", device_extension);
				break;
			}
		}

		if (!enable)
		{
			LOG_WARN("Device extension {} is not supported", device_extension);
		}
	}

	return result;
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

Instance::Instance()
{
	// Initialize volk context
	volkInitialize();

	// Config application info
	VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};

	uint32_t                       sdk_version                = VK_HEADER_VERSION_COMPLETE;
	uint32_t                       api_version                = 0;
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
	std::vector<const char *> extensions_support = GetInstanceExtensionSupported(s_extensions);

	// Config instance info
	VkInstanceCreateInfo create_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	create_info.pApplicationInfo        = &app_info;
	create_info.enabledExtensionCount   = static_cast<uint32_t>(extensions_support.size());
	create_info.ppEnabledExtensionNames = extensions_support.data();
	create_info.enabledLayerCount       = 0;

	// Validation features
	VkValidationFeaturesEXT validation_features{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
	validation_features.enabledValidationFeatureCount = static_cast<uint32_t>(s_validation_extensions.size());
	validation_features.pEnabledValidationFeatures    = s_validation_extensions.data();

	// Enable validation layers
	if (m_debug_enable)
	{
		// Enable validation layer
		if (CheckLayerSupported(s_validation_layers.front()))
		{
			LOG_INFO("Enable validation layer: {}", s_validation_layers.front());
			create_info.enabledLayerCount   = static_cast<uint32_t>(s_validation_layers.size());
			create_info.ppEnabledLayerNames = s_validation_layers.data();
			create_info.pNext               = &validation_features;
		}
		else
		{
			LOG_ERROR("Validation layer was required, but not avaliable, disabling debugging");
			m_debug_enable = false;
		}
	}

	// Create instance
	if (!VK_CHECK(vkCreateInstance(&create_info, nullptr, &m_handle)))
	{
		LOG_ERROR("Failed to create vulkan instance!");
		return;
	}
	else
	{
		// Config to volk
		volkLoadInstance(m_handle);
	}

	// Initialize instance extension functions
	CreateDebugUtilsMessengerEXT          = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(m_handle, "vkCreateDebugUtilsMessengerEXT"));
	DestroyDebugUtilsMessengerEXT         = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(m_handle, "vkDestroyDebugUtilsMessengerEXT"));
	SetDebugUtilsObjectTagEXT             = reinterpret_cast<PFN_vkSetDebugUtilsObjectTagEXT>(vkGetInstanceProcAddr(m_handle, "vkSetDebugUtilsObjectTagEXT"));
	SetDebugUtilsObjectNameEXT            = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(vkGetInstanceProcAddr(m_handle, "vkSetDebugUtilsObjectNameEXT"));
	CmdBeginDebugUtilsLabelEXT            = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(vkGetInstanceProcAddr(m_handle, "vkCmdBeginDebugUtilsLabelEXT"));
	CmdEndDebugUtilsLabelEXT              = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(vkGetInstanceProcAddr(m_handle, "vkCmdEndDebugUtilsLabelEXT"));
	GetPhysicalDeviceMemoryProperties2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2KHR>(vkGetInstanceProcAddr(m_handle, "vkGetPhysicalDeviceMemoryProperties2KHR"));

	if (m_debug_enable)
	{
		VKDebugger::Initialize(m_handle);
	}
}

Instance::~Instance()
{
	if (m_debug_enable)
	{
		VKDebugger::Shutdown(m_handle);
	}

	if (m_handle)
	{
		vkDestroyInstance(m_handle, nullptr);
		m_handle = VK_NULL_HANDLE;
	}
}

Instance::operator const VkInstance &() const
{
	return m_handle;
}

const VkInstance &Instance::GetHandle() const
{
	return m_handle;
}

PhysicalDevice::PhysicalDevice(VkInstance instance)
{
	// Get number of physical devices
	uint32_t physical_device_count = 0;
	vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr);

	// Get all physical devices
	std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
	vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data());

	// Select suitable physical device
	m_handle = SelectPhysicalDevice(physical_devices);

	// Get max MSAA sample count
	m_max_samples_count = GetMaxSampleCount(m_handle);

	// Get physical device properties
	vkGetPhysicalDeviceProperties(m_handle, &m_properties);

	// Get physical device features
	vkGetPhysicalDeviceFeatures(m_handle, &m_features);

	// Get physical device memory properties
	vkGetPhysicalDeviceMemoryProperties(m_handle, &m_memory_properties);
}

PhysicalDevice::operator const VkPhysicalDevice &() const
{
	return m_handle;
}

const VkPhysicalDevice &PhysicalDevice::GetHandle() const
{
	return m_handle;
}

const VkPhysicalDeviceProperties &PhysicalDevice::GetProperties() const
{
	return m_properties;
}

const VkPhysicalDeviceFeatures &PhysicalDevice::GetFeatures() const
{
	return m_features;
}

const VkPhysicalDeviceMemoryProperties &PhysicalDevice::GetMemoryProperties() const
{
	return m_memory_properties;
}

const VkSampleCountFlagBits &PhysicalDevice::GetSampleCount() const
{
	return m_max_samples_count;
}

Surface::Surface(VkPhysicalDevice physical_device)
{
	VkResult result = glfwCreateWindowSurface(RenderContext::GetInstance(), static_cast<GLFWwindow *>(Core::Window::GetInstance().GetHandle()), NULL, &m_handle);

	// Get surface capabilities
	if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, m_handle, &m_capabilities) != VK_SUCCESS)
	{
		LOG_ERROR("Failed to get physical device surface capabilities!");
		return;
	}

	// Get surface format
	uint32_t surface_format_count = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, m_handle, &surface_format_count, nullptr);
	std::vector<VkSurfaceFormatKHR> surface_formats(surface_format_count);
	vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, m_handle, &surface_format_count, surface_formats.data());

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
		vkDestroySurfaceKHR(RenderContext::GetInstance(), m_handle, nullptr);
	}
}

Surface::operator const VkSurfaceKHR &() const
{
	return m_handle;
}

const VkSurfaceKHR &Surface::GetHandle() const
{
	return m_handle;
}

const VkSurfaceCapabilitiesKHR &Surface::GetCapabilities() const
{
	return m_capabilities;
}

const VkSurfaceFormatKHR &Surface::GetFormat() const
{
	return m_format;
}

Device::Device()
{
	m_physical_device = std::make_unique<PhysicalDevice>(RenderContext::GetInstance());
	m_surface         = std::make_unique<Surface>(*m_physical_device);

	// Queue supporting
	uint32_t queue_family_property_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(*m_physical_device, &queue_family_property_count, nullptr);
	std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_property_count);
	vkGetPhysicalDeviceQueueFamilyProperties(*m_physical_device, &queue_family_property_count, queue_family_properties.data());

	m_queues.resize(queue_family_property_count);
	m_queue_index.resize(queue_family_property_count, 0);

	std::optional<uint32_t> graphics_family, compute_family, transfer_family, present_family;

	graphics_family = GetQueueFamilyIndex(queue_family_properties, VK_QUEUE_GRAPHICS_BIT);
	transfer_family = GetQueueFamilyIndex(queue_family_properties, VK_QUEUE_TRANSFER_BIT);
	compute_family  = GetQueueFamilyIndex(queue_family_properties, VK_QUEUE_COMPUTE_BIT);

	for (uint32_t i = 0; i < queue_family_property_count; i++)
	{
		// Check for presentation support
		VkBool32 present_support;
		vkGetPhysicalDeviceSurfaceSupportKHR(*m_physical_device, i, *m_surface, &present_support);

		if (queue_family_properties[i].queueCount > 0 && present_support)
		{
			present_family                       = i;
			m_queue_family[QueueFamily::Present] = i;
			break;
		}
	}

	if (graphics_family.has_value())
	{
		m_queue_family[QueueFamily::Graphics] = graphics_family.value();
		m_support_queues |= VK_QUEUE_GRAPHICS_BIT;
		m_queues[m_queue_family[QueueFamily::Graphics]].resize(queue_family_properties[m_queue_family[QueueFamily::Graphics]].queueCount);
	}

	if (compute_family.has_value())
	{
		m_queue_family[QueueFamily::Compute] = compute_family.value();
		m_support_queues |= VK_QUEUE_COMPUTE_BIT;
		m_queues[m_queue_family[QueueFamily::Compute]].resize(queue_family_properties[m_queue_family[QueueFamily::Compute]].queueCount);
	}

	if (transfer_family.has_value())
	{
		m_queue_family[QueueFamily::Transfer] = transfer_family.value();
		m_support_queues |= VK_QUEUE_TRANSFER_BIT;
		m_queues[m_queue_family[QueueFamily::Transfer]].resize(queue_family_properties[m_queue_family[QueueFamily::Transfer]].queueCount);
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

	if (m_support_queues & VK_QUEUE_GRAPHICS_BIT)
	{
		VkDeviceQueueCreateInfo graphics_queue_create_info = {};
		graphics_queue_create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		graphics_queue_create_info.queueFamilyIndex        = m_queue_family[QueueFamily::Graphics];
		graphics_queue_create_info.queueCount              = queue_family_properties[m_queue_family[QueueFamily::Graphics]].queueCount;
		graphics_queue_create_info.pQueuePriorities        = queue_priorities.data();
		queue_create_infos.emplace_back(graphics_queue_create_info);
	}
	else
	{
		m_queue_family[QueueFamily::Graphics] = 0;
	}

	if (m_support_queues & VK_QUEUE_COMPUTE_BIT &&
	    m_queue_family[QueueFamily::Compute] != m_queue_family[QueueFamily::Graphics])
	{
		VkDeviceQueueCreateInfo compute_queue_create_info = {};
		compute_queue_create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		compute_queue_create_info.queueFamilyIndex        = m_queue_family[QueueFamily::Compute];
		compute_queue_create_info.queueCount              = queue_family_properties[m_queue_family[QueueFamily::Compute]].queueCount;
		compute_queue_create_info.pQueuePriorities        = queue_priorities.data();
		queue_create_infos.emplace_back(compute_queue_create_info);
	}
	else
	{
		m_queue_family[QueueFamily::Compute] = m_queue_family[QueueFamily::Graphics];
	}

	if (m_support_queues & VK_QUEUE_TRANSFER_BIT &&
	    m_queue_family[QueueFamily::Transfer] != m_queue_family[QueueFamily::Graphics] &&
	    m_queue_family[QueueFamily::Transfer] != m_queue_family[QueueFamily::Compute])
	{
		VkDeviceQueueCreateInfo transfer_queue_create_info = {};
		transfer_queue_create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		transfer_queue_create_info.queueFamilyIndex        = m_queue_family[QueueFamily::Transfer];
		transfer_queue_create_info.queueCount              = queue_family_properties[m_queue_family[QueueFamily::Transfer]].queueCount;
		transfer_queue_create_info.pQueuePriorities        = queue_priorities.data();
		queue_create_infos.emplace_back(transfer_queue_create_info);
	}
	else
	{
		m_queue_family[QueueFamily::Transfer] = m_queue_family[QueueFamily::Graphics];
	}

	// Enable logical device features
	auto &physical_device_features = m_physical_device->GetFeatures();

#define ENABLE_DEVICE_FEATURE(feature)                               \
	if (physical_device_features.feature)                            \
	{                                                                \
		m_enabled_features.feature = VK_TRUE;                        \
		LOG_INFO("Physical device feature enable: {}", #feature);    \
	}                                                                \
	else                                                             \
	{                                                                \
		LOG_WARN("Physical device feature not found: {}", #feature); \
	}

	ENABLE_DEVICE_FEATURE(sampleRateShading);
	ENABLE_DEVICE_FEATURE(fillModeNonSolid);
	ENABLE_DEVICE_FEATURE(wideLines);
	ENABLE_DEVICE_FEATURE(samplerAnisotropy);
	ENABLE_DEVICE_FEATURE(textureCompressionBC);
	ENABLE_DEVICE_FEATURE(textureCompressionASTC_LDR);
	ENABLE_DEVICE_FEATURE(textureCompressionETC2);
	ENABLE_DEVICE_FEATURE(vertexPipelineStoresAndAtomics);
	ENABLE_DEVICE_FEATURE(fragmentStoresAndAtomics);
	ENABLE_DEVICE_FEATURE(shaderStorageImageExtendedFormats);
	ENABLE_DEVICE_FEATURE(shaderStorageImageWriteWithoutFormat);
	ENABLE_DEVICE_FEATURE(geometryShader);
	ENABLE_DEVICE_FEATURE(tessellationShader);
	ENABLE_DEVICE_FEATURE(multiViewport);
	ENABLE_DEVICE_FEATURE(imageCubeArray);
	ENABLE_DEVICE_FEATURE(robustBufferAccess);
	ENABLE_DEVICE_FEATURE(multiDrawIndirect);
	ENABLE_DEVICE_FEATURE(drawIndirectFirstInstance);

	// Enable draw indirect count
	VkPhysicalDeviceVulkan12Features vulkan12_features          = {};
	vulkan12_features.sType                                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	vulkan12_features.drawIndirectCount                         = VK_TRUE;
	vulkan12_features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
	vulkan12_features.runtimeDescriptorArray                    = VK_TRUE;
	vulkan12_features.descriptorBindingVariableDescriptorCount  = VK_TRUE;
	vulkan12_features.descriptorBindingPartiallyBound           = VK_TRUE;

	// Get support extensions
	auto support_extensions = GetDeviceExtensionSupport(*m_physical_device, s_extensions);

	// Create device
	VkDeviceCreateInfo device_create_info   = {};
	device_create_info.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	device_create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
	device_create_info.pQueueCreateInfos    = queue_create_infos.data();
	if (!Instance::s_validation_layers.empty())
	{
		device_create_info.enabledLayerCount   = static_cast<uint32_t>(Instance::s_validation_layers.size());
		device_create_info.ppEnabledLayerNames = Instance::s_validation_layers.data();
	}
	device_create_info.enabledExtensionCount   = static_cast<uint32_t>(support_extensions.size());
	device_create_info.ppEnabledExtensionNames = support_extensions.data();
	device_create_info.pEnabledFeatures        = &m_enabled_features;
	device_create_info.pNext                   = &vulkan12_features;

	if (!VK_CHECK(vkCreateDevice(*m_physical_device, &device_create_info, nullptr, &m_handle)))
	{
		LOG_ERROR("Failed to create logical device!");
		return;
	}

	// Volk load context
	volkLoadDevice(m_handle);

	// Get device queues
	for (uint32_t family_index = 0; family_index < m_queues.size(); family_index++)
	{
		for (uint32_t i = 0; i < m_queues[family_index].size(); i++)
		{
			vkGetDeviceQueue(m_handle, family_index, i, &m_queues[family_index][i]);
		}
	}

	// Create Vma allocator
	VmaAllocatorCreateInfo allocator_info = {};
	allocator_info.physicalDevice         = *m_physical_device;
	allocator_info.device                 = m_handle;
	allocator_info.instance               = RenderContext::GetInstance();
	allocator_info.vulkanApiVersion       = VK_API_VERSION_1_2;
	if (!VK_CHECK(vmaCreateAllocator(&allocator_info, &m_allocator)))
	{
		LOG_ERROR("Failed to create vulkan memory allocator");
	}
}

Device ::~Device()
{
	if (VK_CHECK(vkDeviceWaitIdle(m_handle)))
	{
		m_command_pools.clear();
		vmaDestroyAllocator(m_allocator);
		vkDestroyDevice(m_handle, nullptr);
	}
}

Device::operator const VkDevice &() const
{
	return m_handle;
}

const VkDevice &Device::GetHandle() const
{
	return m_handle;
}

const VkPhysicalDeviceFeatures &Device::GetEnabledFeatures() const
{
	return m_enabled_features;
}

const VmaAllocator &Device::GetAllocator() const
{
	return m_allocator;
}

const uint32_t Device::GetQueueFamily(QueueFamily queue) const
{
	return m_queue_family.at(queue);
}

const Surface &Device::GetSurface() const
{
	return *m_surface;
}

const PhysicalDevice &Device::GetPhysicalDevice() const
{
	return *m_physical_device;
}

VkQueue Device::GetQueue(QueueFamily queue)
{
	uint32_t queue_family = 0;
	uint32_t queue_index  = 0;

	queue_family = m_queue_family[queue];
	queue_index  = (++m_queue_index[queue_family]) % m_queues[queue_family].size();
	return m_queues[queue_family][queue_index];
}

std::unique_ptr<CommandBuffer> Device::CreateCommandBuffer(QueueFamily queue)
{
	// Request Command pool
	auto thread_id = std::this_thread::get_id();

	size_t hash = 0;
	Core::HashCombine(hash, static_cast<size_t>(queue));
	Core::HashCombine(hash, thread_id);

	if (m_command_pools.find(hash) == m_command_pools.end())
	{
		m_command_pools.emplace(hash, std::make_unique<CommandPool>(queue, CommandPool::ResetMode::ResetPool, thread_id));
	}

	auto& cmd_pool = *m_command_pools[hash];
	return std::make_unique<CommandBuffer>(cmd_pool);
}

Swapchain::Swapchain(uint32_t width, uint32_t height, bool vsync, Swapchain *old_swapchain) :
    m_width(width), m_height(height), m_vsync(vsync)
{
	auto &surface_format       = RenderContext::GetDevice().GetSurface().GetFormat();
	auto &surface_capabilities = RenderContext::GetDevice().GetSurface().GetCapabilities();
	auto  graphics_family      = RenderContext::GetDevice().GetQueueFamily(QueueFamily::Graphics);
	auto  present_family       = RenderContext::GetDevice().GetQueueFamily(QueueFamily::Present);

	uint32_t present_mode_count;
	vkGetPhysicalDeviceSurfacePresentModesKHR(RenderContext::GetDevice().GetPhysicalDevice(), RenderContext::GetDevice().GetSurface(), &present_mode_count, nullptr);
	std::vector<VkPresentModeKHR> present_modes(present_mode_count);
	vkGetPhysicalDeviceSurfacePresentModesKHR(RenderContext::GetDevice().GetPhysicalDevice(), RenderContext::GetDevice().GetSurface(), &present_mode_count, present_modes.data());

	VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;

	for (const auto &_present_mode : present_modes)
	{
		if (!vsync)
		{
			if (_present_mode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				present_mode = _present_mode;
				break;
			}

			else if (_present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
			{
				present_mode = _present_mode;
			}
		}
		else
		{
			if (_present_mode == VK_PRESENT_MODE_FIFO_KHR)
			{
				present_mode = _present_mode;
			}
		}
	}

	// Get swapchain image count
	auto desired_image_count = surface_capabilities.minImageCount + 1;

	if (surface_capabilities.maxImageCount > 0 && desired_image_count > surface_capabilities.maxImageCount)
	{
		desired_image_count = surface_capabilities.maxImageCount;
	}

	// Get pre transform support
	VkSurfaceTransformFlagBitsKHR pre_transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;

	if (surface_capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
	{
		pre_transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	}
	else
	{
		pre_transform = surface_capabilities.currentTransform;
	}

	// Get composite alpha
	VkCompositeAlphaFlagBitsKHR composite_alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

	const std::vector<VkCompositeAlphaFlagBitsKHR> composite_alpha_flags = {
	    VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
	    VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
	    VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
	    VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR};

	for (const auto &composite_alpha_flag : composite_alpha_flags)
	{
		if (surface_capabilities.supportedCompositeAlpha & composite_alpha_flag)
		{
			composite_alpha = composite_alpha_flag;
			break;
		}
	}

	// Create swapchain
	VkSwapchainCreateInfoKHR swapchain_create_info = {};
	swapchain_create_info.sType                    = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	swapchain_create_info.surface                  = RenderContext::GetDevice().GetSurface();
	swapchain_create_info.minImageCount            = desired_image_count;
	swapchain_create_info.imageFormat              = surface_format.format;
	swapchain_create_info.imageColorSpace          = surface_format.colorSpace;
	swapchain_create_info.imageExtent              = VkExtent2D{m_width, m_height};
	swapchain_create_info.imageArrayLayers         = 1;
	swapchain_create_info.imageUsage               = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	swapchain_create_info.imageSharingMode         = VK_SHARING_MODE_EXCLUSIVE;
	swapchain_create_info.preTransform             = pre_transform;
	swapchain_create_info.compositeAlpha           = composite_alpha;
	swapchain_create_info.presentMode              = present_mode;
	swapchain_create_info.clipped                  = VK_TRUE;
	swapchain_create_info.oldSwapchain             = old_swapchain ? old_swapchain->m_handle : VK_NULL_HANDLE;
	swapchain_create_info.imageUsage |= surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0;
	swapchain_create_info.imageUsage |= surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0;

	if (graphics_family != present_family)
	{
		std::array<uint32_t, 2> queue_family        = {graphics_family, present_family};
		swapchain_create_info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
		swapchain_create_info.queueFamilyIndexCount = static_cast<uint32_t>(queue_family.size());
		swapchain_create_info.pQueueFamilyIndices   = queue_family.data();
	}

	vkCreateSwapchainKHR(RenderContext::GetDevice(), &swapchain_create_info, nullptr, &m_handle);
	vkGetSwapchainImagesKHR(RenderContext::GetDevice(), m_handle, &m_image_count, nullptr);

	//m_back_buffers.reserve(m_image_count);
	//std::vector<VkImage> swapchain_images(m_image_count);
	//m_image_views.resize(m_image_count);

	//vkGetSwapchainImagesKHR(RenderContext::GetDevice(), m_handle, &m_image_count, swapchain_images.data());

	// Convert to VKTexture
	//for (auto &image_handle : swapchain_images)
	//{
	//	m_images.emplace_back(image_handle, m_extent.width, m_extent.height, surface_format.format);
	//}
}

Swapchain::~Swapchain()
{
	if (m_handle)
	{
		vkDestroySwapchainKHR(RenderContext::GetDevice(), m_handle, nullptr);
	}
}

Swapchain::operator const VkSwapchainKHR &() const
{
	return m_handle;
}

const VkSwapchainKHR &Swapchain::GetHandle() const
{
	return m_handle;
}

uint32_t Swapchain::GetCurrentIndex() const
{
	return m_current_index;
}

uint32_t Swapchain::GetImageCount() const
{
	return m_image_count;
}

void Swapchain::AcquireNextImage()
{
}

void Swapchain::Present(VkSemaphore semaphore)
{
}
}        // namespace Ilum::Vulkan