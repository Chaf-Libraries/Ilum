#pragma once

#include <volk.h>

#include <vk_mem_alloc.h>

namespace Ilum
{
class RHIDevice;

class ShaderBindingTableInfo
{
  public:
	ShaderBindingTableInfo(RHIDevice *device, uint32_t handle_count);

	~ShaderBindingTableInfo();

	ShaderBindingTableInfo(const ShaderBindingTableInfo &) = delete;
	ShaderBindingTableInfo &operator=(const ShaderBindingTableInfo &) = delete;
	ShaderBindingTableInfo(ShaderBindingTableInfo &&other)            = delete;
	ShaderBindingTableInfo &operator=(ShaderBindingTableInfo &&other) = delete;

	uint8_t *GetData();

	const VkStridedDeviceAddressRegionKHR *GetHandle() const;

	const VkStridedDeviceAddressRegionKHR *operator&() const;

  private:
	RHIDevice *p_device = nullptr;

	uint32_t m_handle_count = 0;
	uint8_t *m_mapped_data  = nullptr;

	VkStridedDeviceAddressRegionKHR m_handle     = {};
	VkBuffer                        m_buffer     = VK_NULL_HANDLE;
	VmaAllocation                   m_allocation = VK_NULL_HANDLE;
	VkDeviceMemory                  m_memory     = VK_NULL_HANDLE;
};

struct ShaderBindingTable
{
	std::unique_ptr<ShaderBindingTableInfo> raygen   = nullptr;
	std::unique_ptr<ShaderBindingTableInfo> miss     = nullptr;
	std::unique_ptr<ShaderBindingTableInfo> hit      = nullptr;
	std::unique_ptr<ShaderBindingTableInfo> callable = nullptr;
};
}        // namespace Ilum