#include "BoundingBox.hpp"

#include <algorithm>

namespace Ilum::geometry
{
BoundingBox::BoundingBox(const glm::vec3 &_min, const glm::vec3 &_max):
    m_min(_min), m_max(_max)
{
}

BoundingBox::operator bool() const
{
	return m_min.x < m_max.x && m_min.y < m_max.y && m_min.z < m_max.z;
}

void BoundingBox::merge(const glm::vec3 &point)
{
	m_max = glm::max(m_max, point);
	m_min = glm::min(m_min, point);
}

void BoundingBox::merge(const std::vector<glm::vec3> &points)
{
	std::for_each(points.begin(), points.end(), [this](const glm::vec3 &p) { merge(p); });
}

BoundingBox BoundingBox::transform(const glm::mat4 &trans) const
{
	glm::vec3 new_center = trans * glm::vec4(center(), 1.f);
	glm::vec3 old_edge   = scale() * 0.5f;
	glm::vec3 new_edge   = glm::vec3(
        std::fabs(trans[0][0]) * old_edge.x + std::fabs(trans[0][1]) * old_edge.y + std::fabs(trans[0][2]) * old_edge.z,
        std::fabs(trans[1][0]) * old_edge.x + std::fabs(trans[1][1]) * old_edge.y + std::fabs(trans[1][2]) * old_edge.z,
        std::fabs(trans[2][0]) * old_edge.x + std::fabs(trans[2][1]) * old_edge.y + std::fabs(trans[2][2]) * old_edge.z
	);

	return BoundingBox(new_center - new_edge, new_center + new_edge);
}

const glm::vec3 &BoundingBox::min() const
{
	return m_min;
}

const glm::vec3 &BoundingBox::max() const
{
	return m_max;
}

const glm::vec3 BoundingBox::center() const
{
	return (m_max + m_min) / 2.f;
}

const glm::vec3 BoundingBox::scale() const
{
	return m_max - m_min;
}
}        // namespace Ilum::geometry