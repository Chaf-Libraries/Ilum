#pragma once

#include "../Vulkan.hpp"

#include <map>

namespace Ilum::Graphics
{
class Device
{
  public:
	Device(VkInstance instance, VkPhysicalDevice physical_device, VkSurfaceKHR surface);
	~Device();

	operator const VkDevice &() const;

	Device(const Device &) = delete;
	Device &operator=(const Device &) = delete;
	Device(Device &&)                 = delete;
	Device &operator=(Device &&) = delete;

	const VkDevice &                GetHandle() const;
	const VkPhysicalDeviceFeatures &GetEnabledFeatures() const;
	const VmaAllocator &            GetAllocator() const;
	const uint32_t                  GetQueueFamily(QueueFamily family) const;
	VkQueue                         GetQueue(QueueFamily family, uint32_t index = 0) const;

	void WaitIdle() const;

  private:
	VkDevice                 m_handle           = VK_NULL_HANDLE;
	VkPhysicalDeviceFeatures m_enabled_features = {};

	VmaAllocator m_allocator = VK_NULL_HANDLE;

	std::map<QueueFamily, uint32_t>   m_queue_family;
	std::vector<std::vector<VkQueue>> m_queues;
};
}        // namespace Ilum::Graphics