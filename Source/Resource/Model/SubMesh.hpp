#pragma once

#include <glm/glm.hpp>

#include "Material/Material.hpp"

namespace Ilum::Resource
{
struct SubMesh
{
  public:
	uint32_t index = 0;

	glm::mat4 pre_transform = glm::mat4(1.f);

	uint32_t vertices_count = 0;
	uint32_t indices_count  = 0;

	uint32_t vertices_offset = 0;
	uint32_t indices_offset  = 0;

	uint32_t meshlet_offset = 0;
	uint32_t meshlet_count  = 0;

	Material material = {};

	struct
	{
		glm::vec3 min;
		glm::vec3 max;
	} bound;
};
}