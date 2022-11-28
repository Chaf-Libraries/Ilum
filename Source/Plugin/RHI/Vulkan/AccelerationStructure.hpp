#pragma once

#include "Fwd.hpp"

namespace Ilum::Vulkan
{
class Buffer;

class AccelerationStructure : public RHIAccelerationStructure
{
  public:
	AccelerationStructure(RHIDevice *device);

	virtual ~AccelerationStructure() override;

	virtual void Update(RHICommand *cmd_buffer, const TLASDesc &desc) override;

	virtual void Update(RHICommand *cmd_buffer, const BLASDesc &desc) override;

	VkAccelerationStructureKHR GetHandle() const;

	uint64_t GetDeviceAddress() const;

  private:
	void Update(RHICommand *cmd_buffer, const VkAccelerationStructureGeometryKHR &geometry, const VkAccelerationStructureBuildRangeInfoKHR &range_info, VkAccelerationStructureTypeKHR type);
  private:
	VkAccelerationStructureKHR m_handle = VK_NULL_HANDLE;

	// TLAS & BLAS
	std::unique_ptr<Buffer> m_buffer         = nullptr;
	std::unique_ptr<Buffer> m_scratch_buffer = nullptr;

	// Only TLAS
	std::unique_ptr<Buffer> m_instance_buffer = nullptr;

	uint64_t m_device_address = 0;
};
}        // namespace Ilum::Vulkan