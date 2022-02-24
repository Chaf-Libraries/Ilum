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

	void reset();

	virtual void build(VkBuildAccelerationStructureModeKHR mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR) = 0;

  protected:
	VkAccelerationStructureKHR     m_handle = VK_NULL_HANDLE;

	uint64_t m_device_address = 0;
	Buffer   m_buffer;

	std::vector<VkAccelerationStructureGeometryKHR> m_acceleration_structure_geometries;
	std::vector<VkAccelerationStructureBuildRangeInfoKHR> m_acceleration_structure_build_range_infos;
	std::vector<uint32_t>                                 m_primitive_counts;
};
}        // namespace Ilum