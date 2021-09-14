#include "PhysicalDevice.hpp"
#include "Instance.hpp"

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
inline uint32_t score_physical_device(VkPhysicalDevice physical_device)
{
	uint32_t score = 0;

	uint32_t device_extension_properties_count = 0;
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, nullptr);

	std::vector<VkExtensionProperties> extension_properties;
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, extension_properties.data());



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

PhysicalDevice::PhysicalDevice(const Instance &instance) :
    m_instance(instance)
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