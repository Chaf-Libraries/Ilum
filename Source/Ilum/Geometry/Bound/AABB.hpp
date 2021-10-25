#pragma once

#include "Bound.hpp"

namespace Ilum
{
class AABB : public Bound
{
  public:
	AABB() = default;

	~AABB() = default;

	virtual bool valid() const override;

	virtual void add(const glm::vec3 &point) override;

	virtual void add(const std::vector<glm::vec3> &points, const std::vector<uint32_t> &indices = {}) override;

	virtual void transform(const glm::mat4 &matrix) override;

	virtual void reset() override;

	const glm::vec3 &min() const;

	const glm::vec3 &max() const;

	const glm::vec3 center() const;

	const glm::vec3 scale() const;

  private:
	glm::vec3 m_min = glm::vec3(std::numeric_limits<float>::infinity());
	glm::vec3 m_max = glm::vec3(-std::numeric_limits<float>::infinity());
};
}        // namespace Ilum