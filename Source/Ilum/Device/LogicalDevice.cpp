#include "LogicalDevice.hpp"
#include "Instance.hpp"
#include "PhysicalDevice.hpp"
#include "Surface.hpp"

#include "Graphics/GraphicsContext.hpp"

namespace Ilum
{
const std::vector<const char *> LogicalDevice::extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME};

inline const std::vector<const char *> get_device_extension_support(const PhysicalDevice &physical_device, const std::vector<const char *> &extensions)
{
	std::vector<const char *> result;

	uint32_t device_extension_properties_count = 0;
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, nullptr);

	std::vector<VkExtensionProperties> extension_properties(device_extension_properties_count);
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, extension_properties.data());

	for (auto &device_extension : LogicalDevice::extensions)
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

inline std::optional<uint32_t> get_queue_family_index(const std::vector<VkQueueFamilyProperties> &queue_family_properties, VkQueueFlagBits queue_flag)
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

LogicalDevice::LogicalDevice()
{
	// Queue supporting
	uint32_t queue_family_property_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(GraphicsContext::instance()->getPhysicalDevice(), &queue_family_property_count, nullptr);
	std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_property_count);
	vkGetPhysicalDeviceQueueFamilyProperties(GraphicsContext::instance()->getPhysicalDevice(), &queue_family_property_count, queue_family_properties.data());

	std::optional<uint32_t> graphics_family, compute_family, transfer_family, present_family;

	graphics_family = get_queue_family_index(queue_family_properties, VK_QUEUE_GRAPHICS_BIT);
	transfer_family = get_queue_family_index(queue_family_properties, VK_QUEUE_TRANSFER_BIT);
	compute_family  = get_queue_family_index(queue_family_properties, VK_QUEUE_COMPUTE_BIT);

	for (uint32_t i = 0; i < queue_family_property_count; i++)
	{
		// Check for presentation support
		VkBool32 present_support;
		vkGetPhysicalDeviceSurfaceSupportKHR(GraphicsContext::instance()->getPhysicalDevice(), i, GraphicsContext::instance()->getSurface(), &present_support);

		if (queue_family_properties[i].queueCount > 0 && present_support)
		{
			present_family   = i;
			m_present_family = i;
			m_present_queues.resize(queue_family_properties[i].queueCount);
			break;
		}
	}

	if (graphics_family.has_value())
	{
		m_graphics_family = graphics_family.value();
		m_support_queues |= VK_QUEUE_GRAPHICS_BIT;
		m_graphics_queues.resize(queue_family_properties[m_graphics_family].queueCount);
	}

	if (compute_family.has_value())
	{
		m_compute_family = compute_family.value();
		m_support_queues |= VK_QUEUE_COMPUTE_BIT;
		m_compute_queues.resize(queue_family_properties[m_compute_family].queueCount);
	}

	if (transfer_family.has_value())
	{
		m_transfer_family = transfer_family.value();
		m_support_queues |= VK_QUEUE_TRANSFER_BIT;
		m_transfer_queues.resize(queue_family_properties[m_transfer_family].queueCount);
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
	auto &physical_device_features = GraphicsContext::instance()->getPhysicalDevice().getFeatures();

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

	// Enable Ray Tracing Extension
	VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_feature = {};
	acceleration_structure_feature.sType                                            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
	acceleration_structure_feature.accelerationStructure                            = VK_TRUE;

	VkPhysicalDeviceRayTracingPipelineFeaturesKHR raty_tracing_pipeline_feature = {};
	raty_tracing_pipeline_feature.sType                                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
	raty_tracing_pipeline_feature.rayTracingPipeline                            = VK_TRUE;
	raty_tracing_pipeline_feature.pNext                                         = &acceleration_structure_feature;

	// Enable Vulkan 1.2 Features
	VkPhysicalDeviceVulkan12Features vulkan12_features          = {};
	vulkan12_features.sType                                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	vulkan12_features.drawIndirectCount                         = VK_TRUE;
	vulkan12_features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
	vulkan12_features.runtimeDescriptorArray                    = VK_TRUE;
	vulkan12_features.descriptorBindingVariableDescriptorCount  = VK_TRUE;
	vulkan12_features.descriptorBindingPartiallyBound           = VK_TRUE;
	vulkan12_features.bufferDeviceAddress                       = VK_TRUE;
	vulkan12_features.shaderOutputLayer                         = VK_TRUE;
	vulkan12_features.shaderOutputViewportIndex                 = VK_TRUE;
	vulkan12_features.pNext                                     = &raty_tracing_pipeline_feature;

	// Get support extensions
	auto support_extensions = get_device_extension_support(GraphicsContext::instance()->getPhysicalDevice(), extensions);

	// Create device
	VkDeviceCreateInfo device_create_info   = {};
	device_create_info.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	device_create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
	device_create_info.pQueueCreateInfos    = queue_create_infos.data();
	if (!Instance::validation_layers.empty())
	{
		device_create_info.enabledLayerCount   = static_cast<uint32_t>(Instance::validation_layers.size());
		device_create_info.ppEnabledLayerNames = Instance::validation_layers.data();
	}
	device_create_info.enabledExtensionCount   = static_cast<uint32_t>(support_extensions.size());
	device_create_info.ppEnabledExtensionNames = support_extensions.data();
	device_create_info.pEnabledFeatures        = &m_enabled_features;
	device_create_info.pNext                   = &vulkan12_features;

	if (!VK_CHECK(vkCreateDevice(GraphicsContext::instance()->getPhysicalDevice(), &device_create_info, nullptr, &m_handle)))
	{
		VK_ERROR("Failed to create logical device!");
		return;
	}

	// Volk load context
	volkLoadDevice(m_handle);

	// Get device queues
	for (uint32_t i = 0; i < static_cast<uint32_t>(m_graphics_queues.size()); i++)
	{
		vkGetDeviceQueue(m_handle, m_graphics_family, i, &m_graphics_queues[i]);
	}

	for (uint32_t i = 0; i < static_cast<uint32_t>(m_compute_queues.size()); i++)
	{
		vkGetDeviceQueue(m_handle, m_compute_family, i, &m_compute_queues[i]);
	}

	for (uint32_t i = 0; i < static_cast<uint32_t>(m_transfer_queues.size()); i++)
	{
		vkGetDeviceQueue(m_handle, m_transfer_family, i, &m_transfer_queues[i]);
	}

	for (uint32_t i = 0; i < static_cast<uint32_t>(m_present_queues.size()); i++)
	{
		vkGetDeviceQueue(m_handle, m_present_family, i, &m_present_queues[i]);
	}

	// Create Vma allocator
	VmaAllocatorCreateInfo allocator_info = {};
	allocator_info.physicalDevice         = GraphicsContext::instance()->getPhysicalDevice();
	allocator_info.device                 = m_handle;
	allocator_info.instance               = GraphicsContext::instance()->getInstance();
	allocator_info.vulkanApiVersion       = VK_API_VERSION_1_2;
	allocator_info.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (!VK_CHECK(vmaCreateAllocator(&allocator_info, &m_allocator)))
	{
		VK_ERROR("Failed to create vulkan memory allocator");
	}
}

LogicalDevice::~LogicalDevice()
{
	if (VK_CHECK(vkDeviceWaitIdle(m_handle)))
	{
		vmaDestroyAllocator(m_allocator);
		vkDestroyDevice(m_handle, nullptr);
	}
}

LogicalDevice::operator const VkDevice &() const
{
	return m_handle;
}

const VkDevice &LogicalDevice::getLogicalDevice() const
{
	return m_handle;
}

const VkPhysicalDeviceFeatures &LogicalDevice::getEnabledFeatures() const
{
	return m_enabled_features;
}

const VmaAllocator &LogicalDevice::getAllocator() const
{
	return m_allocator;
}

const uint32_t LogicalDevice::getGraphicsFamily() const
{
	return m_graphics_family;
}

const uint32_t LogicalDevice::getComputeFamily() const
{
	return m_compute_family;
}

const uint32_t LogicalDevice::getTransferFamily() const
{
	return m_transfer_family;
}

const uint32_t LogicalDevice::getPresentFamily() const
{
	return m_present_family;
}

const std::vector<VkQueue> &LogicalDevice::getGraphicsQueues() const
{
	return m_graphics_queues;
}

const std::vector<VkQueue> &LogicalDevice::getComputeQueues() const
{
	return m_compute_queues;
}

const std::vector<VkQueue> &LogicalDevice::getTransferQueues() const
{
	return m_transfer_queues;
}

const std::vector<VkQueue> &LogicalDevice::getPresentQueues() const
{
	return m_present_queues;
}

VkQueueFlagBits LogicalDevice::getPresentQueueFlag() const
{
	if (m_present_family == m_graphics_family)
	{
		return VK_QUEUE_GRAPHICS_BIT;
	}
	if (m_present_family == m_compute_family)
	{
		return VK_QUEUE_COMPUTE_BIT;
	}
	if (m_present_family == m_transfer_family)
	{
		return VK_QUEUE_TRANSFER_BIT;
	}
	return VK_QUEUE_GRAPHICS_BIT;
}
}        // namespace Ilum