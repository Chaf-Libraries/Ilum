#include "VKDevice.hpp"
#include "VKCommandPool.hpp"
#include "VKContext.hpp"
#include "VKInstance.hpp"
#include "VKSurface.hpp"

#include <Core/Hash.hpp>

#include <map>
#include <optional>
#include <sstream>
#include <vector>

namespace Ilum::RHI::Vulkan
{
const std::vector<const char *> VKDevice::s_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME};

inline uint32_t ScorePhysicalDevice(VkPhysicalDevice physical_device)
{
	uint32_t score = 0;

	// Check extensions
	uint32_t device_extension_properties_count = 0;
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, nullptr);

	std::vector<VkExtensionProperties> extension_properties(device_extension_properties_count);
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, extension_properties.data());

	for (auto &device_extension : VKDevice::s_extensions)
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

inline const std::vector<const char *> GetDeviceExtensionSupport(const VKPhysicalDevice &physical_device, const std::vector<const char *> &extensions)
{
	std::vector<const char *> result;

	uint32_t device_extension_properties_count = 0;
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, nullptr);

	std::vector<VkExtensionProperties> extension_properties(device_extension_properties_count);
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, extension_properties.data());

	for (auto &device_extension : VKDevice::s_extensions)
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

VKPhysicalDevice::VKPhysicalDevice(VkInstance instance)
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

VKPhysicalDevice::operator const VkPhysicalDevice &() const
{
	return m_handle;
}

const VkPhysicalDevice &VKPhysicalDevice::GetHandle() const
{
	return m_handle;
}

const VkPhysicalDeviceProperties &VKPhysicalDevice::GetProperties() const
{
	return m_properties;
}

const VkPhysicalDeviceFeatures &VKPhysicalDevice::GetFeatures() const
{
	return m_features;
}

const VkPhysicalDeviceMemoryProperties &VKPhysicalDevice::GetMemoryProperties() const
{
	return m_memory_properties;
}

const VkSampleCountFlagBits &VKPhysicalDevice::GetSampleCount() const
{
	return m_max_samples_count;
}

VKDevice::VKDevice()
{
	m_physical_device = std::make_unique<VKPhysicalDevice>(VKContext::GetInstance());
	m_surface         = std::make_unique<VKSurface>(*m_physical_device);

	// Queue supporting
	uint32_t queue_family_property_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(*m_physical_device, &queue_family_property_count, nullptr);
	std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_property_count);
	vkGetPhysicalDeviceQueueFamilyProperties(*m_physical_device, &queue_family_property_count, queue_family_properties.data());

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
			present_family   = i;
			m_present_family = i;
			break;
		}
	}

	if (graphics_family.has_value())
	{
		m_graphics_family = graphics_family.value();
		m_support_queues |= VK_QUEUE_GRAPHICS_BIT;
	}

	if (compute_family.has_value())
	{
		m_compute_family = compute_family.value();
		m_support_queues |= VK_QUEUE_COMPUTE_BIT;
	}

	if (transfer_family.has_value())
	{
		m_transfer_family = transfer_family.value();
		m_support_queues |= VK_QUEUE_TRANSFER_BIT;
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
		graphics_queue_create_info.queueFamilyIndex        = m_graphics_family;
		graphics_queue_create_info.queueCount              = queue_family_properties[m_graphics_family].queueCount;
		graphics_queue_create_info.pQueuePriorities        = queue_priorities.data();
		queue_create_infos.emplace_back(graphics_queue_create_info);
	}
	else
	{
		m_graphics_family = 0;
	}

	if (m_support_queues & VK_QUEUE_COMPUTE_BIT && m_compute_family != m_graphics_family)
	{
		VkDeviceQueueCreateInfo compute_queue_create_info = {};
		compute_queue_create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		compute_queue_create_info.queueFamilyIndex        = m_compute_family;
		compute_queue_create_info.queueCount              = queue_family_properties[m_compute_family].queueCount;
		compute_queue_create_info.pQueuePriorities        = queue_priorities.data();
		queue_create_infos.emplace_back(compute_queue_create_info);
	}
	else
	{
		m_compute_family = m_graphics_family;
	}

	if (m_support_queues & VK_QUEUE_TRANSFER_BIT && m_transfer_family != m_graphics_family && m_transfer_family != m_compute_family)
	{
		VkDeviceQueueCreateInfo transfer_queue_create_info = {};
		transfer_queue_create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		transfer_queue_create_info.queueFamilyIndex        = m_transfer_family;
		transfer_queue_create_info.queueCount              = queue_family_properties[m_transfer_family].queueCount;
		transfer_queue_create_info.pQueuePriorities        = queue_priorities.data();
		queue_create_infos.emplace_back(transfer_queue_create_info);
	}
	else
	{
		m_transfer_family = m_graphics_family;
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
	if (!VKInstance::s_validation_layers.empty())
	{
		device_create_info.enabledLayerCount   = static_cast<uint32_t>(VKInstance::s_validation_layers.size());
		device_create_info.ppEnabledLayerNames = VKInstance::s_validation_layers.data();
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

	// Create Vma allocator
	VmaAllocatorCreateInfo allocator_info = {};
	allocator_info.physicalDevice         = *m_physical_device;
	allocator_info.device                 = m_handle;
	allocator_info.instance               = VKContext::GetInstance();
	allocator_info.vulkanApiVersion       = VK_API_VERSION_1_2;
	if (!VK_CHECK(vmaCreateAllocator(&allocator_info, &m_allocator)))
	{
		LOG_ERROR("Failed to create vulkan memory allocator");
	}
}

VKDevice::~VKDevice()
{
	m_command_pools.clear();

	if (VK_CHECK(vkDeviceWaitIdle(m_handle)))
	{
		vmaDestroyAllocator(m_allocator);
		vkDestroyDevice(m_handle, nullptr);
	}
}

VKDevice::operator const VkDevice &() const
{
	return m_handle;
}

const VkDevice &VKDevice::GetHandle() const
{
	return m_handle;
}

const VkPhysicalDeviceFeatures &VKDevice::GetEnabledFeatures() const
{
	return m_enabled_features;
}

const VmaAllocator &VKDevice::GetAllocator() const
{
	return m_allocator;
}

const uint32_t VKDevice::GetGraphicsFamily() const
{
	return m_graphics_family;
}

const uint32_t VKDevice::GetComputeFamily() const
{
	return m_compute_family;
}

const uint32_t VKDevice::GetTransferFamily() const
{
	return m_transfer_family;
}

const uint32_t VKDevice::GetPresentFamily() const
{
	return m_present_family;
}

VKCommandPool &VKDevice::AcquireCommandPool(const CmdUsage &usage, const std::thread::id &thread_id)
{
	// Hash id
	size_t hash_id = 0;
	Core::HashCombine(hash_id, static_cast<size_t>(usage));
	Core::HashCombine(hash_id, thread_id);

	if (m_command_pools.find(hash_id) == m_command_pools.end())
	{
		m_command_pools.emplace(hash_id, std::make_unique<VKCommandPool>(usage, thread_id));
	}

	return *m_command_pools[hash_id];
}

VKCommandPool &VKDevice::GetCommandPool(size_t pool_index)
{
	assert(m_command_pools.find(pool_index) != m_command_pools.end());
	return *m_command_pools[pool_index];
}

bool VKDevice::HasCommandPool(size_t pool_index)
{
	return m_command_pools.find(pool_index) != m_command_pools.end();
}
}        // namespace Ilum::RHI::Vulkan