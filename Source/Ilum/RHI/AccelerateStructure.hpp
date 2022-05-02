#pragma once

#include <volk.h>

#include <vk_mem_alloc.h>

#include <string>
#include <memory>

namespace Ilum
{
class RHIDevice;
class Buffer;

struct AccelerationStructureDesc
{
	VkAccelerationStructureTypeKHR           type;
	VkAccelerationStructureGeometryKHR       geometry;
	VkAccelerationStructureBuildRangeInfoKHR range_info;
};

class AccelerationStructure
{
  public:
	AccelerationStructure(RHIDevice *device);
	~AccelerationStructure();

	AccelerationStructure(const AccelerationStructure &) = delete;
	AccelerationStructure &operator=(const AccelerationStructure &) = delete;
	AccelerationStructure(AccelerationStructure &&other)            = delete;
	AccelerationStructure &operator=(AccelerationStructure &&other) = delete;

	uint64_t GetDeviceAddress() const;

	operator VkAccelerationStructureKHR() const;

	void Build(VkCommandBuffer cmd_buffer, AccelerationStructureDesc desc);

	void SetName(const std::string &name);

  private:
	RHIDevice                 *p_device = nullptr;
	VkAccelerationStructureKHR m_handle = VK_NULL_HANDLE;
	uint64_t                   m_device_address = 0;
	std::unique_ptr<Buffer>    m_buffer         = nullptr;
};
using AccelerationStructureReference= std::reference_wrapper<AccelerationStructure>;
}        // namespace Ilum