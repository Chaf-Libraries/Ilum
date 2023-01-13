#pragma once

#include <glm/glm.hpp>

namespace Ilum
{
struct Meshlet
{
	glm::vec3 center;
	float     radius;
	
	glm::vec3 cone_axis;
	float    cone_cutoff;

	uint32_t data_offset;
	uint32_t vertex_offset;
	uint32_t vertex_count;
	uint32_t triangle_count;

	template<typename Archive>
	void serialize(Archive& archive)
	{
		archive(center, radius, cone_axis, cone_cutoff, data_offset, vertex_offset, vertex_count, triangle_count);
	}
};
}        // namespace Ilum