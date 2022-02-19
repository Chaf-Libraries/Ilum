#pragma once

#include "Graphics/Vulkan.hpp"

namespace Ilum::Graphics
{
class Device;

class ShaderBindingTable
{
  public:
	ShaderBindingTable(const Device &device, uint32_t group_count);
	~ShaderBindingTable();

	const VkStridedDeviceAddressRegionKHR *GetStridedDeviceAddressRegion() const;
	uint8_t *GetData() const;

  private:
	const Device &                  m_device;
	VkStridedDeviceAddressRegionKHR m_strided_device_address_region = {};
	uint64_t                        m_device_address                = 0;
	VkBuffer                        m_handle                        = VK_NULL_HANDLE;
	VmaAllocation                   m_allocation                    = VK_NULL_HANDLE;
	uint8_t *                       m_mapped_data                   = nullptr;
};
}        // namespace Ilum::Graphics