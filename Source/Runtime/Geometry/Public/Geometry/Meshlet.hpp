#pragma once

#include <glm/glm.hpp>

namespace Ilum
{
struct alignas(16) Meshlet
{
	glm::vec3 center;
	float     radius;
	int8_t    cone_axis[3];
	int8_t    cone_cutoff;

	uint32_t data_offset;
	uint8_t  vertex_count;
	uint8_t  triangle_count;
};
}        // namespace Ilum