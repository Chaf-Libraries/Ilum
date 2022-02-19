#pragma once

#include "Surface.hpp"

namespace Ilum::geometry
{
struct BSplineSurface : public Surface
{
	virtual void generateVertices(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices, const std::vector<std::vector<glm::vec3>> &control_points, uint32_t sample_x, uint32_t sample_y) override;

	virtual glm::vec3 value(const std::vector<std::vector<glm::vec3>> &control_points, float u, float v) override;

	uint32_t order = 0;
};
}        // namespace Ilum::geometry