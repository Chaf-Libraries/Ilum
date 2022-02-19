#pragma once

#include "Meshlet.hpp"
#include "SubMesh.hpp"
#include "Vertex.hpp"

#include <vector>

namespace Ilum::Resource
{
struct Model
{
	std::vector<SubMesh> submeshes;

	uint32_t vertices_count  = 0;
	uint32_t vertices_offset = 0;

	uint32_t indices_count  = 0;
	uint32_t indices_offset = 0;

	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;
	std::vector<Meshlet>  meshlets;

	struct
	{
		glm::vec3 min;
		glm::vec3 max;
	} bound;
};

using ModelReference = std::reference_wrapper<Model>;
}        // namespace Ilum::Resource