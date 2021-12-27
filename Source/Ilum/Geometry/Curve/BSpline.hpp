#pragma once

#include "Curve.hpp"

namespace Ilum::geometry
{
struct BSpline : public Curve
{
	virtual std::vector<glm::vec3> generateVertices(const std::vector<glm::vec3> &control_points, uint32_t sample) override;

	virtual glm::vec3 value(const std::vector<glm::vec3> &control_points, float t) override;

  private:
	void genDeBoorPoints(const std::vector<glm::vec3> &control_points);

	std::vector<glm::vec3> m_de_boor;
	std::vector<float>     m_knots;
};
}        // namespace Ilum::geometry