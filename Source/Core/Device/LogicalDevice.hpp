#pragma once

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
class Instance;
class PhysicalDevice;
class Surface;

class LogicalDevice
{
  public:
	LogicalDevice(const Surface &surface);

	~LogicalDevice();

	operator const VkDevice &() const;

	const VkDevice &getLogicalDevice() const;

	const VkPhysicalDeviceFeatures &getEnabledFeatures() const;

	const VmaAllocator &getAllocator() const;

	const uint32_t getGraphicsFamily() const;

	const uint32_t getComputeFamily() const;

	const uint32_t getTransferFamily() const;

	const uint32_t getPresentFamily() const;

	const std::vector<VkQueue>& getGraphicsQueues() const;

	const std::vector<VkQueue>& getComputeQueues() const;

	const std::vector<VkQueue>& getTransferQueues() const;

	const std::vector<VkQueue>& getPresentQueues() const;

  public:
	// Extensions
	static const std::vector<const char *> extensions;

  private:
	const Instance &      m_instance;
	const PhysicalDevice &m_physical_device;
	const Surface &       m_surface;

	VkDevice                 m_handle           = VK_NULL_HANDLE;
	VkPhysicalDeviceFeatures m_enabled_features = {};

	VmaAllocator m_allocator = VK_NULL_HANDLE;

	VkQueueFlags m_support_queues  = {};
	uint32_t     m_graphics_family = 0;
	uint32_t     m_compute_family  = 0;
	uint32_t     m_transfer_family = 0;
	uint32_t     m_present_family  = 0;

	std::vector<VkQueue> m_graphics_queues;
	std::vector<VkQueue> m_present_queues;
	std::vector<VkQueue> m_compute_queues;
	std::vector<VkQueue> m_transfer_queues;
};
}        // namespace Ilum