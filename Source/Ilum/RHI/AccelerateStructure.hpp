#pragma once

#include "Geometry/AABB.hpp"

#include <Shaders/ShaderInterop.hpp>

#include <volk.h>

#include <vk_mem_alloc.h>

#include <memory>
#include <string>
#include <vector>

namespace Ilum
{
class RHIDevice;
class Buffer;
class Mesh;

struct AccelerationStructureDesc
{
	VkAccelerationStructureTypeKHR           type;
	VkAccelerationStructureGeometryKHR       geometry;
	VkAccelerationStructureBuildRangeInfoKHR range_info;
};

struct TLASDesc
{
	AABB aabb;

	std::vector<std::pair<glm::mat4, Mesh *>> mesh_instances;
};

struct BLASDesc
{
	Mesh *mesh = nullptr;
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

	void Build(const TLASDesc &desc);
	void Build(const BLASDesc &desc);

	void SetName(const std::string &name);

	// Fall back layer
	Buffer &GetHierarchyBuffer();
	Buffer &GetBoundingVolumeBuffer();
	Buffer &GetPrimitiveIndicesBuffer();

  private:
	void Build(VkCommandBuffer cmd_buffer, AccelerationStructureDesc desc);

	// Temp
  public:
	std::vector<ShaderInterop::HierarchyNode> hierarchy_buffer;

	std::vector<AABB> bvhs;

  private:
	RHIDevice                 *p_device         = nullptr;
	VkAccelerationStructureKHR m_handle         = VK_NULL_HANDLE;
	uint64_t                   m_device_address = 0;
	std::unique_ptr<Buffer>    m_buffer         = nullptr;
	std::unique_ptr<Buffer>    m_scratch_buffer = nullptr;
	// TLAS
	std::unique_ptr<Buffer> m_instance_buffer = nullptr;

	// Fall back layer
	std::unique_ptr<Buffer> m_hierarchy_buffer = nullptr;
	std::unique_ptr<Buffer> m_aabbs_buffer = nullptr;
	std::unique_ptr<Buffer> m_primitive_indices_buffer = nullptr;
};
using AccelerationStructureReference = std::reference_wrapper<AccelerationStructure>;
}        // namespace Ilum