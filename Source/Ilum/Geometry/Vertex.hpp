#pragma once

#include <glm/glm.hpp>

namespace Ilum
{
struct Vertex
{
	alignas(16) glm::vec3 position  = {};
	alignas(16) glm::vec2 texcoord  = {};
	alignas(16) glm::vec3 normal    = {};
	alignas(16) glm::vec3 tangent   = {};
	alignas(16) glm::vec3 bitangent = {};

	Vertex() = default;

	Vertex(glm::vec3 position, glm::vec2 texcoord, glm::vec3 normal, glm::vec3 tangent, glm::vec3 bitangent) :
	    position(position),
	    texcoord(texcoord),
	    normal(normal),
	    tangent(tangent),
	    bitangent(bitangent)
	{
	}
};
}        // namespace Ilum