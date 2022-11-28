#pragma once

#include <glm/glm.hpp>

namespace Ilum
{
struct Meshlet
{
	uint32_t vertices_offset;
	uint32_t indices_offset;
	uint32_t vertices_count;
	uint32_t indices_count;

	glm::vec3 min_bound;
	uint32_t  meshlet_vertices_offset;

	glm::vec3 max_bound;
	uint32_t  meshlet_indices_offset;
};
}        // namespace Ilum