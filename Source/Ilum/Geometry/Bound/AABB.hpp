#pragma once

#include <glm/glm.hpp>

#include <vector>

namespace Ilum
{
struct AABB
{
  public:
	glm::vec3 min = glm::vec3(std::numeric_limits<float>::max());
	glm::vec3 max = glm::vec3(-std::numeric_limits<float>::min());

  public:
	AABB() = default;

	AABB(const glm::vec3 &min, const glm::vec3 &max);

	~AABB() = default;

	void Merge(const glm::vec3 &point);

	void Merge(const std::vector<glm::vec3> &points);

	void Merge(const AABB &aabb);

	AABB Transform(const glm::mat4 &transform);

	const glm::vec3 Center() const;

	const glm::vec3 Scale() const;
};
}        // namespace Ilum