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

	void merge(const BoundingBox &bounding_box);

	BoundingBox transform(const glm::mat4 &trans) const;

	const glm::vec3 center() const;

	const glm::vec3 scale() const;

	bool isInside(const glm::vec3 &point) const;

	bool valid() const;

	glm::vec3 min_ = glm::vec3(std::numeric_limits<float>::max());
	glm::vec3 max_ = glm::vec3(-std::numeric_limits<float>::min());
};
}        // namespace Ilum::geometry