#pragma once

#include "AccelerationStructure.hpp"

namespace Ilum
{
// Bottom Level Acceleration Structure
class BLAS : public AccelerationStructure
{
  public:
	BLAS()  = default;
	~BLAS() = default;

	void add(const VkAccelerationStructureGeometryTrianglesDataKHR &triangle_data);
	void add(const VkAccelerationStructureBuildRangeInfoKHR &range_info);

	virtual void build(VkBuildAccelerationStructureModeKHR mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR) override;
};
}        // namespace Ilum