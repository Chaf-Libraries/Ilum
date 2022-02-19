#pragma once

#include "../Vertex.hpp"

#include <vector>

#include <glm/glm.hpp>

namespace Ilum::geometry
{
struct Surface
{
	virtual void generateVertices(std::vector<Vertex> &vertices, std::vector<uint32_t>& indices, const std::vector<std::vector<glm::vec3>> &control_points, uint32_t sample_x, uint32_t sample_y) = 0;

	virtual glm::vec3 value(const std::vector<std::vector<glm::vec3>> &control_points, float u, float v) = 0;
};
}