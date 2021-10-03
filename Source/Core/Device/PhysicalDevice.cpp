#include "PhysicalDevice.hpp"
#include "Instance.hpp"
#include "LogicalDevice.hpp"

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
inline uint32_t score_physical_device(VkPhysicalDevice physical_device)
{
	uint32_t score = 0;

	// Check extensions
	uint32_t device_extension_properties_count = 0;
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, nullptr);

	std::vector<VkExtensionProperties> extension_properties(device_extension_properties_count);
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, extension_properties.data());

	for (auto &device_extension : LogicalDevice::extensions)
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
	ss<<"\nFound physical device ["<<properties.deviceID<<"]\n";
	ss<<"Name: "<<properties.deviceName<<"\n";
	ss<<"Type: ";
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
		ss << "Other " << properties.deviceType <<"\n";
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
		VK_VERSION_PATCH(properties.apiVersion)
	};

	ss << "API Version: " << supportedVersion[0] << "." << supportedVersion[1] << "." << supportedVersion[2] << '\n';
	
	ss <<"Extensions: \n";
	for (auto& extension : extension_properties)
	{
		ss<<"           "<<extension.extensionName<<"\n";
	}

	VK_INFO("{}", ss.str());

	// Score discrete gpu
	if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
	{
		score += 1000;
	}

	score += properties.limits.maxImageDimension2D;
	return score;
}

inline VkPhysicalDevice select_physical_device(const std::vector<VkPhysicalDevice> &physical_devices)
{
	// Score - GPU
	std::map<uint32_t, VkPhysicalDevice, std::greater<uint32_t>> scores;
	for (auto &gpu : physical_devices)
	{
		scores.emplace(score_physical_device(gpu), gpu);
	}

	if (scores.empty())
	{
		return VK_NULL_HANDLE;
	}

	return scores.begin()->second;
}

inline VkSampleCountFlagBits get_max_sample_count(VkPhysicalDevice physical_device)
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

PhysicalDevice::PhysicalDevice(const Instance &instance)
{
	// Get number of physical devices
	uint32_t physical_device_count = 0;
	vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr);

	// Get all physical devices
	std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
	vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data());

	// Select suitable physical device
	m_handle = select_physical_device(physical_devices);

	// Get max MSAA sample count
	m_max_samples_count = get_max_sample_count(m_handle);

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

const VkPhysicalDevice &PhysicalDevice::getPhysicalDevice() const
{
	return m_handle;
}

const VkPhysicalDeviceProperties &PhysicalDevice::getProperties() const
{
	return m_properties;
}

const VkPhysicalDeviceFeatures &PhysicalDevice::getFeatures() const
{
	return m_features;
}

const VkPhysicalDeviceMemoryProperties &PhysicalDevice::getMemoryProperties() const
{
	return m_memory_properties;
}

const VkSampleCountFlagBits &PhysicalDevice::getSampleCount() const
{
	return m_max_samples_count;
}
}        // namespace Ilum