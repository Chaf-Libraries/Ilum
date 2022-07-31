#pragma once

#include "RHIDevice.hpp"

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

	virtual bool IsRayTracingSupport() override;
	virtual bool IsMeshShaderSupport() override;
	virtual bool IsBufferDeviceAddressSupport() override;
	virtual bool IsBindlessResourceSupport() override;

	VkInstance       GetInstance() const;
	VkPhysicalDevice GetPhysicalDevice() const;
	VkDevice         GetDevice() const;
	VmaAllocator     GetAllocator() const;

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
};
}        // namespace Ilum::Vulkan