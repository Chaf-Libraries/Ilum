#pragma once

#include "SubMesh.hpp"

#include "Graphics/Buffer/Buffer.h"

#include "Geometry/BoundingBox.hpp"

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
	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;

	// Meshlet, for mesh shading & cluster culling
	std::vector<Meshlet>  meshlets;
	//std::vector<meshopt_Meshlet> meshlets;
	//std::vector<meshopt_Bounds>  meshlet_bounds;
	//std::vector<uint32_t>    meshlet_vertices;
	//std::vector<uint8_t>   meshlet_triangles;

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