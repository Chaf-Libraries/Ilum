#pragma once

#include <vector>

#include <glm/glm.hpp>

namespace Ilum::Geo
{
class Bound
{
  public:
	Bound() = default;

	Bound(const glm::vec3 &_min, const glm::vec3 &_max);

	~Bound() = default;

	operator bool() const;

	void Merge(const glm::vec3 &point);

	void Merge(const std::vector<glm::vec3> &points);

	void Merge(const Bound &bounding_box);

	Bound Transform(const glm::mat4 &trans) const;

	const glm::vec3 Center() const;

	const glm::vec3 Scale() const;

	const glm::vec3 &GetMin() const;

	const glm::vec3 &GetMax() const;

	bool IsInside(const glm::vec3 &point) const;
  private:
	glm::vec3 m_min = glm::vec3(std::numeric_limits<float>::max());
	glm::vec3 m_max = glm::vec3(-std::numeric_limits<float>::min());
};
}        // namespace Ilum::Geo