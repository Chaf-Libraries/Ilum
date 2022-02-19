#pragma once

#include "Curve.hpp"

namespace Ilum::geometry
{
struct BSpline : public Curve
{
	virtual std::vector<glm::vec3> generateVertices(const std::vector<glm::vec3> &control_points, uint32_t sample) override;

	virtual glm::vec3 value(const std::vector<glm::vec3> &control_points, float t) override;

	uint32_t order = 0;
};
}        // namespace Ilum::geometry