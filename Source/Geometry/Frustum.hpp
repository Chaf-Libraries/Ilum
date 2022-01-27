#pragma once

#include <array>

#include "Plane.hpp"

namespace Ilum::Geo
{
class Bound;

class Frustum
{
  public:
	Frustum() = default;

	Frustum(const glm::mat4 &view_projection);

	~Frustum() = default;

	bool IsInside(const glm::vec3 &p);

	bool IsInside(const Bound &bbox);

  private:
	std::array<Plane, 6> m_planes;
};
}        // namespace Ilum::Geo