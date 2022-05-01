#pragma once

#include <volk.h>

#include <vk_mem_alloc.h>

namespace Ilum
{
class RHIDevice;

class ShaderBindingTable
{
  public:
	ShaderBindingTable(RHIDevice *device, uint32_t handle_count);

	~ShaderBindingTable();

	ShaderBindingTable(const ShaderBindingTable &) = delete;
	ShaderBindingTable &operator=(const ShaderBindingTable &) = delete;
	ShaderBindingTable(ShaderBindingTable &&other)            = delete;
	ShaderBindingTable &operator=(ShaderBindingTable &&other) = delete;

	uint8_t *GetData();

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
}        // namespace Ilum