#include "AccelerationStructure.hpp"

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/GraphicsContext.hpp"

#include "Device/LogicalDevice.hpp"

namespace Ilum
{
AccelerationStructure::~AccelerationStructure()
{
	if (m_handle)
	{
		vkDestroyAccelerationStructureKHR(GraphicsContext::instance()->getLogicalDevice(), m_handle, nullptr);
	}
}

const VkAccelerationStructureKHR AccelerationStructure::operator&() const
{
	return m_handle;
}

const VkAccelerationStructureKHR &AccelerationStructure::getHandle() const
{
	return m_handle;
}

uint64_t AccelerationStructure::getDeviceAddress() const
{
	return m_device_address;
}

const Buffer &AccelerationStructure::getBuffer() const
{
	return m_buffer;
}

void AccelerationStructure::reset()
{
	m_acceleration_structure_geometries.clear();
	m_acceleration_structure_build_range_infos.clear();
}
}        // namespace Ilum