#pragma once

#include <glm/glm.hpp>

namespace Ilum::Geo
{
struct Vertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 texcoord;
};
}