#pragma once

#include "SubMesh.hpp"

#include <Graphics/Resource/Buffer.hpp>

#include "Geometry/BoundingBox.hpp"
#include "Geometry/Mesh/TriMesh.hpp"

#include <meshoptimizer.h>

namespace Ilum
{
struct Meshlet
{
	meshopt_Bounds bounds;
	uint32_t       indices_offset;
	uint32_t       indices_count;
	uint32_t       vertices_offset;
};

struct Model
{
  public:
	std::vector<SubMesh> submeshes;

	uint32_t vertices_count = 0;
	uint32_t indices_count  = 0;

	uint32_t vertices_offset = 0;
	uint32_t indices_offset  = 0;

	// Raw geometry, original data
	geometry::TriMesh mesh;

	// Meshlet, for mesh shading & cluster culling
	std::vector<Meshlet>  meshlets;

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