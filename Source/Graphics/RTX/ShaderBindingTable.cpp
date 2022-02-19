#include "ShaderBindingTable.hpp"

namespace Ilum::Graphics
{
ShaderBindingTable::ShaderBindingTable(const Device &device, uint32_t group_count):
    m_device(device)
{
}

ShaderBindingTable::~ShaderBindingTable()
{
}

const VkStridedDeviceAddressRegionKHR *ShaderBindingTable::GetStridedDeviceAddressRegion() const
{
	return &m_strided_device_address_region;
}

uint8_t *ShaderBindingTable::GetData() const
{
	return m_mapped_data;
}
}        // namespace Ilum::Graphics