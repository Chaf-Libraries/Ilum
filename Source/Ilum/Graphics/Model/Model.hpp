#pragma once

#include "SubMesh.hpp"

#include "Graphics/Buffer/Buffer.h"

#include "Geometry/BoundingBox.hpp"

namespace Ilum
{
struct Model
{
  public:
	std::vector<SubMesh> submeshes;

	uint32_t vertices_count = 0;
	uint32_t indices_count  = 0;

	uint32_t vertices_offset = 0;
	uint32_t indices_offset  = 0;

	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;

	geometry::BoundingBox bounding_box;

	Model() = default;

	~Model() = default;

	Model(const Model &) = delete;

	Model &operator=(const Model &) = delete;

	Model(Model &&other) noexcept;

	Model &operator=(Model &&other) noexcept;
};

using ModelReference = std::reference_wrapper<Model>;
}        // namespace Ilum