#pragma once

#include <glm/glm.hpp>

#include <vector>

namespace Ilum
{
class BoundingBox
{
  public:
	BoundingBox() = default;

	BoundingBox(const glm::vec3 &_min, const glm::vec3 &_max);

	~BoundingBox() = default;

	operator bool() const;

	void Merge(const glm::vec3 &point);

	void Merge(const std::vector<glm::vec3> &points);

	void Merge(const BoundingBox &bounding_box);

	BoundingBox Transform(const glm::mat4 &trans) const;

	const glm::vec3 &GetMin() const;

	const glm::vec3 &GetMax() const;

	const glm::vec3 Center() const;

	const glm::vec3 Scale() const;

	bool IsInside(const glm::vec3 &point) const;

	bool IsValid() const;

  private:
	glm::vec3 m_min = glm::vec3(std::numeric_limits<float>::max());
	glm::vec3 m_max = glm::vec3(-std::numeric_limits<float>::min());
};
}        // namespace Ilum