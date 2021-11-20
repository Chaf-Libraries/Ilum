#include "Plane.hpp"

namespace Ilum::geometry
{
Plane::Plane(const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2):
    normal(glm::normalize(glm::cross(p1 - p0, p2 - p0))),
    constant(-glm::dot(normal, p0))
{

}

Plane::Plane(const glm::vec3 &normal, const glm::vec3 &point)
{
	this->normal = glm::normalize(normal);
	this->constant = -glm::dot(normal, point);
}

Plane::Plane(const glm::vec3 &normal, float distance):
    normal(glm::normalize(normal)), constant(distance)
{
}

Plane Plane::transform(const glm::mat4 &trans) const
{
	glm::vec4 new_plane = glm::transpose(glm::inverse(trans)) * glm::vec4(normal, constant);
	return Plane(glm::normalize(glm::vec4(new_plane)), new_plane.w);
}

glm::vec3 Plane::relect(const glm::vec3 &direction) const
{
	return direction - (2.f * glm::dot(normal, direction) * normal);
}
}        // namespace Ilum::geometry