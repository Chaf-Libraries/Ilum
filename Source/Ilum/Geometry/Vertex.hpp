#pragma once

#include <glm/glm.hpp>

namespace Ilum
{
struct Vertex
{
	glm::vec3 position;
	glm::vec2 texcoord;
	glm::vec3 normal;
	glm::vec3 tangent;
};
}