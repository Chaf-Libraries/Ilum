#pragma once

#include <glm/glm.hpp>

#include <vector>

namespace Ilum
{
class AABB
{
  public:
	AABB() = default;

	AABB(const glm::vec3 &_min, const glm::vec3 &_max);

	~AABB() = default;

	operator bool() const;

	void Merge(const glm::vec3 &point);

	void Merge(const std::vector<glm::vec3> &points);

	void Merge(const AABB &aabb);

	void Merge(const std::vector<AABB> &aabbs);

	AABB Transform(const glm::mat4 &trans) const;

	const glm::vec3 &GetMin() const;

	const glm::vec3 &GetMax() const;

	const glm::vec3 Center() const;

	const glm::vec3 Scale() const;

	bool IsInside(const glm::vec3 &point) const;

	bool IsValid() const;

	void Reset();

  private:
	glm::vec3 m_min = glm::vec3(std::numeric_limits<float>::max());
	glm::vec3 m_max = glm::vec3(-std::numeric_limits<float>::max());
};
}        // namespace Ilum