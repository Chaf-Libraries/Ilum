#pragma once

#include "Curve.hpp"

namespace Ilum::geometry
{
// Bezier Spline with uniform parameterization and natural end condition
struct BezierSpline : public Curve
{
	// New generated control points
	std::vector<glm::vec3> control_points;

	virtual std::vector<glm::vec3> generateVertices(const std::vector<glm::vec3> &control_points, uint32_t sample) override;

	virtual glm::vec3 value(const std::vector<glm::vec3> &control_points, float t) override;

  private:
	void generateControlPoints(const std::vector<glm::vec3> &control_points);
};
}        // namespace Ilum::geometry