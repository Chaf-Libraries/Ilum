#pragma once

#include <glm/glm.hpp>

namespace Ilum::Geo
{
class Bound;

class Ray
{
  public:
	Ray() = default;

	Ray(const glm::vec3 &origin, const glm::vec3 &direction);

	~Ray() = default;

	// Project a point on the ray
	glm::vec3 Project(const glm::vec3 &point) const;

	float Distance(const glm::vec3 &point) const;

	float Hit(const Bound &bbox);

  private:
	glm::vec3 m_origin;
	glm::vec3 m_direction;
};
}        // namespace Ilum::Geo