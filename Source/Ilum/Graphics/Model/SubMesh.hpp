#pragma once

#include "Geometry/BoundingBox.hpp"
#include "Geometry/Mesh/TriMesh.hpp"
#include "Vertex.hpp"

#include "Material/DisneyPBR.h"

namespace Ilum
{
struct SubMesh
{
  public:
	uint32_t index_offset = 0;

	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;
	material::DisneyPBR             material;

	geometry::BoundingBox bounding_box;

	bool visible = true;

	SubMesh(std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices, uint32_t index_offset, scope<material::DisneyPBR> &&material = nullptr);

	~SubMesh() = default;

	SubMesh(const SubMesh &) = delete;

	SubMesh &operator=(const SubMesh &) = delete;

	SubMesh(SubMesh &&other) noexcept;

	SubMesh &operator=(SubMesh &&other) noexcept;
};
}        // namespace Ilum