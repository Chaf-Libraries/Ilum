#pragma once

#include <array>

#include "Plane.hpp"

namespace Ilum
{
class BoundingBox;

class Frustum
{
  public:
	Frustum() = default;

	Frustum(const glm::mat4 &view_projection);

	~Frustum() = default;

	bool IsInside(const glm::vec3 &p);

	bool IsInside(const BoundingBox &bbox);

	const std::array<Plane, 6> &GetPlanes() const;

  private:
	std::array<Plane, 6> planes;
};
}        // namespace Ilum