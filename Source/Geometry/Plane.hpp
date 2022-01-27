#pragma once

#include <glm/glm.hpp>

namespace Ilum::Geo
{
class Plane
{
  public:
	Plane() = default;

	Plane(const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2);

	Plane(const glm::vec3 &normal, const glm::vec3 &point);

	Plane(const glm::vec3 &normal, float distance);

	~Plane() = default;

	const glm::vec3 &GetNormal() const;

	float GetConstant() const;

	Plane Transform(const glm::mat4 &trans) const;

	glm::vec3 Reflect(const glm::vec3 &direction) const;

	float Distance(const glm::vec3 &p) const;

  private:
	// Plane equation: Ax+By+Cz+D=0
	glm::vec3 m_normal   = glm::vec3(0.f);
	float     m_constant = 0;
};
}        // namespace Ilum::Geo