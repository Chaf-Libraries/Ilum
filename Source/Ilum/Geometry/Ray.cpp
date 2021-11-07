#include "Ray.hpp"

namespace Ilum::geometry
{
Ray::Ray(const glm::vec3 &origin, const glm::vec3 &direction) :
    origin(origin), direction(direction)
{
}

glm::vec3 Ray::project(const glm::vec3 &point) const
{
	return origin + glm::dot((point - origin), direction) * direction;
}

float Ray::distance(const glm::vec3 &point) const
{
	return (point-project(point)).length();
}
}        // namespace Ilum::geometry