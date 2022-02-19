#pragma once

#include <glm/glm.hpp>

namespace Ilum::geometry
{
struct BoundingBox;

struct Ray
{
	Ray() = default;

	Ray(const glm::vec3 &origin, const glm::vec3 &direction);

	~Ray() = default;

	// Project a point on the ray
	glm::vec3 project(const glm::vec3 &point) const;

	float distance(const glm::vec3 &point) const;

	float hit(const BoundingBox &bbox);

	glm::vec3 origin;
	glm::vec3 direction;
};
}        // namespace Ilum::geometry