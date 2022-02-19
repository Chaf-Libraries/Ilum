#include "Device.hpp"

#include <optional>

namespace Ilum::Graphics
{
inline const std::vector<const char *> GetDeviceExtensionSupport(const VkPhysicalDevice &physical_device, const std::vector<const char *> &extensions)
{
	std::vector<const char *> result;

	uint32_t device_extension_properties_count = 0;
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, nullptr);

	std::vector<VkExtensionProperties> extension_properties(device_extension_properties_count);
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, extension_properties.data());

	for (auto &device_extension : DeviceExtension::extensions)
	{
		bool enable = false;
		for (auto &support_extension : extension_properties)
		{
			if (std::strcmp(device_extension, support_extension.extensionName) == 0)
			{
				result.push_back(device_extension);
				enable = true;
				VK_INFO("Enable device extension: {}", device_extension);
				break;
			}
		}

		if (!enable)
		{
			VK_WARN("Device extension {} is not supported", device_extension);
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

Device::Device(VkInstance instance, VkPhysicalDevice physical_device, VkSurfaceKHR surface)
{
	// Queue supporting
	uint32_t queue_family_property_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_property_count, nullptr);
	std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_property_count);
	vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_property_count, queue_family_properties.data());

	std::optional<uint32_t> graphics_family, compute_family, transfer_family, present_family;
	m_queues.resize(4);

	graphics_family = GetQueueFamilyIndex(queue_family_properties, VK_QUEUE_GRAPHICS_BIT);
	transfer_family = GetQueueFamilyIndex(queue_family_properties, VK_QUEUE_TRANSFER_BIT);
	compute_family  = GetQueueFamilyIndex(queue_family_properties, VK_QUEUE_COMPUTE_BIT);

	for (uint32_t i = 0; i < queue_family_property_count; i++)
	{
		// Check for presentation support
		VkBool32 present_support;
		vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, surface, &present_support);

		if (queue_family_properties[i].queueCount > 0 && present_support)
		{
			present_family                       = i;
			m_queue_family[QueueFamily::Present] = i;
			m_queues[m_queue_family[QueueFamily::Present]].resize(queue_family_properties[i].queueCount);
			break;
		}
	}

	if (graphics_family.has_value())
	{
		m_queue_family[QueueFamily::Graphics] = graphics_family.value();
		m_queues[m_queue_family[QueueFamily::Graphics]].resize(queue_family_properties[graphics_family.value()].queueCount);
	}

	if (compute_family.has_value())
	{
		m_queue_family[QueueFamily::Compute] = compute_family.value();
		m_queues[m_queue_family[QueueFamily::Compute]].resize(queue_family_properties[compute_family.value()].queueCount);
	}

	if (transfer_family.has_value())
	{
		m_queue_family[QueueFamily::Transfer] = transfer_family.value();
		m_queues[m_queue_family[QueueFamily::Transfer]].resize(queue_family_properties[transfer_family.value()].queueCount);
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

	if (m_queue_family.find(QueueFamily::Graphics) != m_queue_family.end())
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

	if (m_queue_family.find(QueueFamily::Compute) != m_queue_family.end() && m_queue_family[QueueFamily::Compute] != m_queue_family[QueueFamily::Graphics])
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

	if (m_queue_family.find(QueueFamily::Transfer) != m_queue_family.end() && m_queue_family[QueueFamily::Transfer] != m_queue_family[QueueFamily::Graphics] && m_queue_family[QueueFamily::Transfer] != m_queue_family[QueueFamily::Compute])
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
	VkPhysicalDeviceFeatures physical_device_features = {};
	vkGetPhysicalDeviceFeatures(physical_device, &physical_device_features);

#define ENABLE_DEVICE_FEATURE(feature)                             \
	if (physical_device_features.feature)                          \
	{                                                              \
		m_enabled_features.feature = VK_TRUE;                      \
		VK_INFO("Physical device feature enable: {}", #feature)    \
	}                                                              \
	else                                                           \
	{                                                              \
		VK_WARN("Physical device feature not found: {}", #feature) \
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
	auto support_extensions = GetDeviceExtensionSupport(physical_device, DeviceExtension::extensions);

	// Create device
	VkDeviceCreateInfo device_create_info   = {};
	device_create_info.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	device_create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
	device_create_info.pQueueCreateInfos    = queue_create_infos.data();
	if (!InstanceExtension::validation_layers.empty())
	{
		device_create_info.enabledLayerCount   = static_cast<uint32_t>(InstanceExtension::validation_layers.size());
		device_create_info.ppEnabledLayerNames = InstanceExtension::validation_layers.data();
	}
	device_create_info.enabledExtensionCount   = static_cast<uint32_t>(support_extensions.size());
	device_create_info.ppEnabledExtensionNames = support_extensions.data();
	device_create_info.pEnabledFeatures        = &m_enabled_features;
	device_create_info.pNext                   = &vulkan12_features;

	if (!VK_CHECK(vkCreateDevice(physical_device, &device_create_info, nullptr, &m_handle)))
	{
		VK_ERROR("Failed to create logical device!");
		return;
	}

	// Volk load context
	volkLoadDevice(m_handle);

	// Get device queues
	for (uint32_t i = 0; i < static_cast<uint32_t>(m_queues[m_queue_family[QueueFamily::Graphics]].size()); i++)
	{
		vkGetDeviceQueue(m_handle, m_queue_family[QueueFamily::Graphics], i, &m_queues[m_queue_family[QueueFamily::Graphics]][i]);
	}

	for (uint32_t i = 0; i < static_cast<uint32_t>(m_queues[m_queue_family[QueueFamily::Compute]].size()); i++)
	{
		vkGetDeviceQueue(m_handle, m_queue_family[QueueFamily::Compute], i, &m_queues[m_queue_family[QueueFamily::Compute]][i]);
	}

	for (uint32_t i = 0; i < static_cast<uint32_t>(m_queues[m_queue_family[QueueFamily::Transfer]].size()); i++)
	{
		vkGetDeviceQueue(m_handle, m_queue_family[QueueFamily::Transfer], i, &m_queues[m_queue_family[QueueFamily::Transfer]][i]);
	}

	for (uint32_t i = 0; i < static_cast<uint32_t>(m_queues[m_queue_family[QueueFamily::Present]].size()); i++)
	{
		vkGetDeviceQueue(m_handle, m_queue_family[QueueFamily::Present], i, &m_queues[m_queue_family[QueueFamily::Present]][i]);
	}

	// Create Vma allocator
	VmaAllocatorCreateInfo allocator_info = {};
	allocator_info.physicalDevice         = physical_device;
	allocator_info.device                 = m_handle;
	allocator_info.instance               = instance;
	allocator_info.vulkanApiVersion       = VK_API_VERSION_1_2;
	if (!VK_CHECK(vmaCreateAllocator(&allocator_info, &m_allocator)))
	{
		VK_ERROR("Failed to create vulkan memory allocator");
	}
}

Device::~Device()
{
	if (VK_CHECK(vkDeviceWaitIdle(m_handle)))
	{
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

const uint32_t Device::GetQueueFamily(QueueFamily family) const
{
	return m_queue_family.at(family);
}

VkQueue Device::GetQueue(QueueFamily family, uint32_t index) const
{
	return m_queues[m_queue_family.at(family)][index];
}
}        // namespace Ilum::Graphics