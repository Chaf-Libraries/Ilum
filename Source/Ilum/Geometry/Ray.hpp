#pragma once

#include <glm/glm.hpp>

namespace Ilum::geometry
{
struct AABB;

struct Ray
{
	Ray(const glm::vec3 &origin, const glm::vec3 &direction);

	~Ray() = default;

	// Project a point on the ray
	glm::vec3 project(const glm::vec3 &point) const;

	float distance(const glm::vec3 &point) const;

	glm::vec3 origin;
	glm::vec3 direction;
};
}        // namespace Ilum::geometry