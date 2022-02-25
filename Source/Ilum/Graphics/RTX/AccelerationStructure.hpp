#pragma once

#include "Graphics/Buffer/Buffer.h"
#include "Graphics/Vulkan/Vulkan.hpp"

namespace Ilum
{
class AccelerationStructure
{
  public:
	AccelerationStructure() = default;

	~AccelerationStructure();

	const VkAccelerationStructureKHR operator&() const;

	const VkAccelerationStructureKHR &getHandle() const;

	uint64_t getDeviceAddress() const;

	const Buffer &getBuffer() const;

	void build(VkAccelerationStructureGeometryKHR &geometry, VkAccelerationStructureBuildRangeInfoKHR &range_info, VkAccelerationStructureTypeKHR type);

  protected:
	VkAccelerationStructureKHR     m_handle = VK_NULL_HANDLE;

	uint64_t m_device_address = 0;
	Buffer   m_buffer;
};
}        // namespace Ilum