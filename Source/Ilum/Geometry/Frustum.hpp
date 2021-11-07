#pragma once

#include <array>

#include "Plane.hpp"

namespace Ilum::geometry
{
struct Frustum
{
	std::array<Plane, 6> planes;

	Frustum() = default;

	Frustum(const glm::mat4 &view_projection);

	~Frustum() = default;
};
}        // namespace Ilum::geometry