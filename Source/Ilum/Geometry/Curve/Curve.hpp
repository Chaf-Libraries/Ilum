#pragma once

#include <vector>

#include <glm/glm.hpp>

namespace Ilum::geometry
{
struct Curve
{
	virtual std::vector<glm::vec3> generateVertices(const std::vector<glm::vec3> &control_points, uint32_t sample)
	{
		return {};
	}

	virtual glm::vec3 value(const std::vector<glm::vec3> &control_points, float t) = 0;
};
}        // namespace Ilum::geometry