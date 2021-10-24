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
	m_min = glm::max(m_min, point);
}

void AABB::add(const std::vector<glm::vec3> &points, const std::vector<uint32_t> &indices)
{
}

void AABB::transform(const glm::mat4 &matrix)
{
}
}        // namespace Ilum