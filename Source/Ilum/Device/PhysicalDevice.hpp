#pragma once

#include "Graphics/Vulkan/Vulkan.hpp"

namespace Ilum
{
class PhysicalDevice
{
  public:
	PhysicalDevice();

	operator const VkPhysicalDevice &() const;

	const VkPhysicalDevice &getPhysicalDevice() const;

	const VkPhysicalDeviceProperties &getProperties() const;

	const VkPhysicalDeviceRayTracingPipelinePropertiesKHR &getRayTracingPipelineProperties() const;

	const VkPhysicalDeviceFeatures &getFeatures() const;

	const VkPhysicalDeviceMemoryProperties &getMemoryProperties() const;

	const VkSampleCountFlagBits &getSampleCount() const;

  private:
	VkPhysicalDevice                                m_handle                         = VK_NULL_HANDLE;
	VkPhysicalDeviceProperties                      m_properties                     = {};
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_raytracing_pipeline_properties = {};
	VkPhysicalDeviceFeatures                        m_features                       = {};
	VkPhysicalDeviceMemoryProperties                m_memory_properties              = {};
	VkSampleCountFlagBits                           m_max_samples_count              = VK_SAMPLE_COUNT_1_BIT;
};
}        // namespace Ilum