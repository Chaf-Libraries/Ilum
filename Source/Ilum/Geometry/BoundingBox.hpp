#pragma once

#include <glm/glm.hpp>

#include <vector>

namespace Ilum::geometry
{
// AABB Bounding box
struct BoundingBox
{
	BoundingBox() = default;

	BoundingBox(const glm::vec3 &_min, const glm::vec3 &_max);

	~BoundingBox() = default;

	operator bool() const;

	void merge(const glm::vec3 &point);

	void merge(const std::vector<glm::vec3> &points);

	BoundingBox transform(const glm::mat4 &trans) const;

	const glm::vec3 &min() const;

	const glm::vec3 &max() const;

	const glm::vec3 center() const;

	const glm::vec3 scale() const;

	glm::vec3 m_min = glm::vec3(std::numeric_limits<float>::infinity());
	glm::vec3 m_max = glm::vec3(-std::numeric_limits<float>::infinity());
};
}        // namespace Ilum::geometry