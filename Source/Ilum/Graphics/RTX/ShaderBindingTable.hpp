#pragma once

#include <Graphics/Buffer/Buffer.h>

namespace Ilum
{
class ShaderBindingTable
{
  public:
	ShaderBindingTable(uint32_t handle_count);

	~ShaderBindingTable();

	ShaderBindingTable(const ShaderBindingTable &) = delete;

	ShaderBindingTable &operator=(const ShaderBindingTable &) = delete;

	ShaderBindingTable(ShaderBindingTable &&other);

	ShaderBindingTable &operator=(ShaderBindingTable &&other);

	uint8_t *getData();

	const VkStridedDeviceAddressRegionKHR *getHandle() const;

	const VkStridedDeviceAddressRegionKHR *operator&() const;

  private:
	uint32_t m_handle_count = 0;

	VkStridedDeviceAddressRegionKHR m_handle = {};

	VkBuffer m_buffer = VK_NULL_HANDLE;

	VmaAllocation m_allocation = VK_NULL_HANDLE;

	VkDeviceMemory m_memory = VK_NULL_HANDLE;

	uint8_t *m_mapped_data = nullptr;
};
}        // namespace Ilum