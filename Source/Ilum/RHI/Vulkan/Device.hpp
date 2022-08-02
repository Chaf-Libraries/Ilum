#pragma once

#include "RHIDevice.hpp"
#include "Definitions.hpp"

#include <volk.h>

#include <vk_mem_alloc.h>

namespace Ilum::Vulkan
{
class Device : public RHIDevice
{
  private:
	void CreateInstance();
	void CreatePhysicalDevice();
	void CreateLogicalDevice();

  public:
	Device();
	virtual ~Device() override;

	virtual void WaitIdle() override;

	virtual bool IsRayTracingSupport() override;
	virtual bool IsMeshShaderSupport() override;
	virtual bool IsBufferDeviceAddressSupport() override;
	virtual bool IsBindlessResourceSupport() override;

	VkInstance       GetInstance() const;
	VkPhysicalDevice GetPhysicalDevice() const;
	VkDevice         GetDevice() const;
	VmaAllocator     GetAllocator() const;

	uint32_t GetQueueFamily(RHIQueueFamily family);
	uint32_t GetQueueCount(RHIQueueFamily family);

	VkCommandPool AcquireCommandPool(uint32_t frame_index, RHIQueueFamily family);
	void          ResetCommandPool(uint32_t frame_index);

  private:
	// Supported extensions
	std::vector<const char *> m_supported_instance_extensions;
	std::vector<const char *> m_supported_device_features;
	std::vector<const char *> m_supported_device_extensions;

  private:
	VkInstance       m_instance        = VK_NULL_HANDLE;
	VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
	VkDevice         m_logical_device  = VK_NULL_HANDLE;
	VmaAllocator     m_allocator       = VK_NULL_HANDLE;

	// Queue Family
	uint32_t m_graphics_family = 0;
	uint32_t m_compute_family  = 0;
	uint32_t m_transfer_family = 0;

	uint32_t m_graphics_queue_count = 0;
	uint32_t m_compute_queue_count = 0;
	uint32_t m_transfer_queue_count = 0;

	// Command Pool
	std::vector<std::unordered_map<std::thread::id, VkCommandPool>> m_cmd_pools;
};
}        // namespace Ilum::Vulkan