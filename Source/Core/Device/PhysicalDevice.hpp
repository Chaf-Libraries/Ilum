#pragma once

#include "Core/Engine/Vulkan/Vulkan.hpp"

namespace Ilum
{
class Instance;

class PhysicalDevice
{
  public:
	PhysicalDevice(const Instance &instance);

	const Instance &getInstance() const;

	operator const VkPhysicalDevice &() const;

	const VkPhysicalDevice &                getPhysicalDevice() const;

	const VkPhysicalDeviceProperties &      getProperties() const;

	const VkPhysicalDeviceFeatures &        getFeatures() const;

	const VkPhysicalDeviceMemoryProperties &getMemoryProperties() const;

	const VkSampleCountFlagBits &           getSampleCount() const;

  private:
	const Instance &m_instance;

	VkPhysicalDevice                 m_handle            = VK_NULL_HANDLE;
	VkPhysicalDeviceProperties       m_properties        = {};
	VkPhysicalDeviceFeatures         m_features          = {};
	VkPhysicalDeviceMemoryProperties m_memory_properties = {};
	VkSampleCountFlagBits            m_max_samples_count    = VK_SAMPLE_COUNT_1_BIT;
};
}        // namespace Ilum