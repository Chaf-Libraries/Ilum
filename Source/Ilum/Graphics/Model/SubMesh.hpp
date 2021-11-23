#pragma once

#include "Geometry/BoundingBox.hpp"
#include "Geometry/Mesh/TriMesh.hpp"
#include "Vertex.hpp"

#include "Material/DisneyPBR.h"

namespace Ilum
{
class CommandBuffer;

struct SubMesh
{
  public:
	uint32_t index = 0;

	glm::mat4 pre_transform = glm::mat4(1.f);

	uint32_t vertices_count = 0;
	uint32_t indices_count  = 0;

	uint32_t vertices_offset = 0;
	uint32_t indices_offset  = 0;

	material::DisneyPBR material;

	geometry::BoundingBox bounding_box;

	VkDrawIndexedIndirectCommand indirect_cmd = {};

	SubMesh() = default;

	~SubMesh() = default;

	SubMesh(const SubMesh &) = delete;

	SubMesh &operator=(const SubMesh &) = delete;

	SubMesh(SubMesh &&other) noexcept;

	SubMesh &operator=(SubMesh &&other) noexcept;
};
}        // namespace Ilum