#pragma once

#include <glm/glm.hpp>

#include <vector>

namespace Ilum
{
class AABB
{
  public:
	AABB() = default;

	~AABB() = default;

	virtual bool valid() const;

	virtual void add(const glm::vec3 &point);

	virtual void add(const std::vector<glm::vec3> &points, const std::vector<uint32_t> &indices = {});

	virtual void reset();

	const glm::vec3 &min() const;

	const glm::vec3 &max() const;

	const glm::vec3 center() const;

	const glm::vec3 scale() const;

  private:
	glm::vec3 m_min = glm::vec3(std::numeric_limits<float>::infinity());
	glm::vec3 m_max = glm::vec3(-std::numeric_limits<float>::infinity());
};
}        // namespace Ilum