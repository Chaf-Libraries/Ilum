#pragma once

#include <glm/glm.hpp>

namespace Ilum::Geo
{
struct Vertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 texcoord;

	Vertex() = default;

	Vertex(const glm::vec3& position, const glm::vec3& normal, const glm::vec2& texcoord) :
		position(position), normal(normal), texcoord(texcoord)
	{

	}
};
}