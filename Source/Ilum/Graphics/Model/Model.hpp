#pragma once

#include "SubMesh.hpp"

#include "Graphics/Buffer/Buffer.h"

#include "Geometry/BoundingBox.hpp"

#include <meshoptimizer.h>

namespace Ilum
{
struct Meshlet
{
	glm::vec3 center;
	float     radius;

	glm::vec3 cone_apex;
	float     cone_cutoff;

	uint32_t indices_offset;
	uint32_t indices_count;
	uint32_t vertices_offset;
	uint32_t vertices_count;

	alignas(16)glm::vec3 cone_axis;
};

struct Model
{
  public:
	std::vector<SubMesh> submeshes;

	uint32_t vertices_count = 0;
	uint32_t indices_count  = 0;
	uint32_t meshlet_count  = 0;

	uint32_t vertices_offset = 0;
	uint32_t indices_offset  = 0;
	uint32_t meshlet_offset  = 0;

	Buffer meshlets_buffer;
	Buffer vertices_buffer;
	Buffer indices_buffer;

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