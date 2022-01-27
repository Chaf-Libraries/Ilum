#include "Plane.hpp"

namespace Ilum::Geo
{
Plane::Plane(const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2) :
    m_normal(glm::normalize(glm::cross(p1 - p0, p2 - p0))),
    m_constant(-glm::dot(m_normal, p0))
{
}

Plane::Plane(const glm::vec3 &normal, const glm::vec3 &point)
{
	this->m_normal   = glm::normalize(normal);
	this->m_constant = -glm::dot(normal, point);
}

Plane::Plane(const glm::vec3 &normal, float distance) :
    m_normal(glm::normalize(normal)), m_constant(distance)
{
}

const glm::vec3 &Plane::GetNormal() const
{
	return m_normal;
}

float Plane::GetConstant() const
{
	return m_constant;
}

Plane Plane::Transform(const glm::mat4 &trans) const
{
	glm::vec4 new_plane = glm::transpose(glm::inverse(trans)) * glm::vec4(m_normal, m_constant);
	return Plane(glm::normalize(glm::vec4(new_plane)), new_plane.w);
}

glm::vec3 Plane::Reflect(const glm::vec3 &direction) const
{
	return direction - (2.f * glm::dot(m_normal, direction) * m_normal);
}

float Plane::Distance(const glm::vec3 &p) const
{
	return std::fabs(glm::dot(m_normal, p) + m_constant) / glm::length(m_normal);
}
}