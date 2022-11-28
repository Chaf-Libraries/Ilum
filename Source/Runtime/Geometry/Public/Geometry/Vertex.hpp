#pragma once

#include <glm/glm.hpp>

namespace Ilum
{
struct Vertex
{
	alignas(16) glm::vec3 position;
	alignas(16) glm::vec3 normal;
	alignas(16) glm::vec3 tangent;
	glm::vec2 uv0;
	glm::vec2 uv1;
};
}        // namespace Ilum