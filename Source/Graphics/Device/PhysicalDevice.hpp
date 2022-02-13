#pragma once

#include "../Vulkan.hpp"

namespace Ilum::Graphics
{
class Instance;

class PhysicalDevice
{
  public:
	PhysicalDevice(const Instance &instance);
	~PhysicalDevice() = default;

	PhysicalDevice(const PhysicalDevice &) = delete;
	PhysicalDevice &operator=(const PhysicalDevice &) = delete;
	PhysicalDevice(PhysicalDevice &&)                 = delete;
	PhysicalDevice &operator=(PhysicalDevice &&) = delete;

	operator const VkPhysicalDevice &() const;

	const VkPhysicalDevice &                GetHandle() const;
	const VkPhysicalDeviceProperties &      GetProperties() const;
	const VkPhysicalDeviceFeatures &        GetFeatures() const;
	const VkPhysicalDeviceMemoryProperties &GetMemoryProperties() const;
	const VkSampleCountFlagBits &           GetSampleCount() const;

  private:
	VkPhysicalDevice                 m_handle            = VK_NULL_HANDLE;
	VkPhysicalDeviceProperties       m_properties        = {};
	VkPhysicalDeviceFeatures         m_features          = {};
	VkPhysicalDeviceMemoryProperties m_memory_properties = {};
	VkSampleCountFlagBits            m_max_samples_count = VK_SAMPLE_COUNT_1_BIT;
};
}        // namespace Ilum::Graphics