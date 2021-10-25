#include "AABB.hpp"

namespace Ilum
{
bool AABB::valid() const
{
	return m_min.x < m_max.x && m_min.y < m_max.y && m_min.z < m_max.z;
}

void AABB::add(const glm::vec3 &point)
{
	m_max = glm::max(m_max, point);
	m_min = glm::min(m_min, point);
}

void AABB::add(const std::vector<glm::vec3> &points, const std::vector<uint32_t> &indices)
{
	if (indices.empty())
	{
		std::for_each(points.begin(), points.end(), [this](const glm::vec3 &p) { add(p); });
	}
	else
	{
		std::for_each(indices.begin(), indices.end(), [this, &points](const uint32_t index) { add(points[index]); });
	}
}

void AABB::transform(const glm::mat4 &matrix)
{
	glm::vec3 min_ = m_min;
	glm::vec3 max_ = m_max;

	reset();

	add(matrix * glm::vec4(min_.x, min_.y, max_.z, 1.0f));
	add(matrix * glm::vec4(min_.x, max_.y, min_.z, 1.0f));
	add(matrix * glm::vec4(min_.x, max_.y, max_.z, 1.0f));
	add(matrix * glm::vec4(max_.x, min_.y, min_.z, 1.0f));
	add(matrix * glm::vec4(max_.x, min_.y, max_.z, 1.0f));
	add(matrix * glm::vec4(max_.x, max_.y, min_.z, 1.0f));
	add(matrix * glm::vec4(max_, 1.0f));
	add(matrix * glm::vec4(min_, 1.0f));
}

void AABB::reset()
{
	m_min = glm::vec3(std::numeric_limits<float>::infinity());
	m_max = glm::vec3(-std::numeric_limits<float>::infinity());
}

const glm::vec3 &AABB::min() const
{
	return m_min;
}

const glm::vec3 &AABB::max() const
{
	return m_max;
}

const glm::vec3 AABB::center() const
{
	return (m_max + m_min) / 2.f;
}

const glm::vec3 AABB::scale() const
{
	return m_max - m_min;
}
}        // namespace Ilum