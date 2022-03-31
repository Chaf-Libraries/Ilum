#pragma once

#include "Material.hpp"

#include "Geometry/BoundingBox.hpp"
#include "Geometry/Vertex.hpp"

#include "Graphics/RTX/AccelerationStructure.hpp"

namespace Ilum
{
class CommandBuffer;

struct SubMesh
{
  public:
	std::string name = "";

	uint32_t index = 0;

	glm::mat4 pre_transform = glm::mat4(1.f);

	uint32_t vertices_count = 0;
	uint32_t indices_count  = 0;

	uint32_t vertices_offset = 0;
	uint32_t indices_offset  = 0;

	uint32_t meshlet_offset = 0;
	uint32_t meshlet_count  = 0;

	Material material;

	geometry::BoundingBox bounding_box;

	AccelerationStructure bottom_level_as = AccelerationStructure(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);

	SubMesh() = default;

	~SubMesh() = default;

	SubMesh(const SubMesh &) = delete;

	SubMesh &operator=(const SubMesh &) = delete;

	SubMesh(SubMesh &&other) noexcept;

	SubMesh &operator=(SubMesh &&other) noexcept;
};
}        // namespace Ilum