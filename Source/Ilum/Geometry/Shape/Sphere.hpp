#pragma once

#include "Shape.hpp"

#include <glm/glm.hpp>

#include <vector>

namespace Ilum::geometry
{
struct Sphere : public Shape
{
	glm::vec3 center = glm::vec3(0.f);
	float     radius = std::numeric_limits<float>::infinity();

	Sphere() = default;

	Sphere(const glm::vec3 &center, float radius);

	~Sphere() = default;

	void merge(const glm::vec3 &point);

	void merge(const std::vector<glm::vec3> &points);

	std::pair<std::vector<Vertex>, std::vector<uint32_t>> toMesh() override;
};
}        // namespace Ilum::geometry