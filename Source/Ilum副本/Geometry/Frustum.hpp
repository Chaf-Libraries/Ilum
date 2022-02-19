#pragma once

#include <array>

#include "Plane.hpp"

namespace Ilum::geometry
{
struct BoundingBox;

struct Frustum
{
	std::array<Plane, 6> planes;

	Frustum() = default;

	Frustum(const glm::mat4 &view_projection);

	~Frustum() = default;

	bool isInside(const glm::vec3& p);

	bool isInside(const BoundingBox &bbox);
};
}        // namespace Ilum::geometry