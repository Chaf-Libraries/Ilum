#pragma once

#include "Graphics/Buffer/Buffer.h"
#include "Graphics/Vulkan/Vulkan.hpp"

namespace Ilum
{
class AccelerationStructure
{
  public:
	AccelerationStructure(VkAccelerationStructureTypeKHR type);

	~AccelerationStructure();

	AccelerationStructure(const AccelerationStructure &) = delete;

	AccelerationStructure &operator=(const AccelerationStructure &) = delete;

	AccelerationStructure(AccelerationStructure &&other);

	AccelerationStructure &operator=(AccelerationStructure &&other);

	const VkAccelerationStructureKHR operator&() const;

	const VkAccelerationStructureKHR &getHandle() const;

	uint64_t getDeviceAddress() const;

	const Buffer &getBuffer() const;

	// If rebuild as, return true
	bool build(VkAccelerationStructureGeometryKHR &geometry, VkAccelerationStructureBuildRangeInfoKHR &range_info);

  protected:
	VkAccelerationStructureKHR     m_handle = VK_NULL_HANDLE;
	VkAccelerationStructureTypeKHR m_type   = {};

	uint64_t m_device_address = 0;
	Buffer   m_buffer;
};

using AccelerationStructureReference = std::reference_wrapper<const AccelerationStructure>;
}        // namespace Ilum