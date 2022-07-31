#include "AABB.hpp"

namespace Ilum
{
AABB::AABB(const glm::vec3 &_min, const glm::vec3 &_max) :
    m_max(_max), m_min(_min)
{
}

AABB::operator bool() const
{
	return m_min.x < m_max.x && m_min.y < m_max.y && m_min.z < m_max.z;
}

void AABB::Merge(const glm::vec3 &point)
{
	m_max = glm::max(m_max, point);
	m_min = glm::min(m_min, point);
}

void AABB::Merge(const std::vector<glm::vec3> &points)
{
	std::for_each(points.begin(), points.end(), [this](const glm::vec3 &p) { Merge(p); });
}

void AABB::Merge(const AABB &aabb)
{
	m_min.x = aabb.m_min.x < m_min.x ? aabb.m_min.x : m_min.x;
	m_min.y = aabb.m_min.y < m_min.y ? aabb.m_min.y : m_min.y;
	m_min.z = aabb.m_min.z < m_min.z ? aabb.m_min.z : m_min.z;
	m_max.x = aabb.m_max.x > m_max.x ? aabb.m_max.x : m_max.x;
	m_max.y = aabb.m_max.y > m_max.y ? aabb.m_max.y : m_max.y;
	m_max.z = aabb.m_max.z > m_max.z ? aabb.m_max.z : m_max.z;
}

void AABB::Merge(const std::vector<AABB> &aabbs)
{
	for (auto& aabb : aabbs)
	{
		Merge(aabb);
	}
}

AABB AABB::Transform(const glm::mat4 &trans) const
{
	glm::vec3 v[2], xa, xb, ya, yb, za, zb;

	xa = trans[0] * m_min[0];
	xb = trans[0] * m_max[0];

	ya = trans[1] * m_min[1];
	yb = trans[1] * m_max[1];

	za = trans[2] * m_min[2];
	zb = trans[2] * m_max[2];

	v[0] = trans[3];
	v[0] += glm::min(xa, xb);
	v[0] += glm::min(ya, yb);
	v[0] += glm::min(za, zb);

	v[1] = trans[3];
	v[1] += glm::max(xa, xb);
	v[1] += glm::max(ya, yb);
	v[1] += glm::max(za, zb);

	return AABB(v[0], v[1]);
}

const glm::vec3 &AABB::GetMin() const
{
	return m_min;
}

const glm::vec3 &AABB::GetMax() const
{
	return m_max;
}

const glm::vec3 AABB::Center() const
{
	return (m_max + m_min) / 2.f;
}

const glm::vec3 AABB::Scale() const
{
	return m_max - m_min;
}

bool AABB::IsInside(const glm::vec3 &point) const
{
	return !(point.x < m_min.x || point.x > m_max.x || point.y < m_min.y || point.y > m_max.y || point.z < m_min.z || point.z > m_max.z);
}

bool AABB::IsValid() const
{
	return m_min.x <= m_max.x && m_min.y <= m_max.y && m_min.z <= m_max.z;
}

void AABB::Reset()
{
	m_min = glm::vec3(std::numeric_limits<float>::max());
	m_max = glm::vec3(-std::numeric_limits<float>::min());
}

}        // namespace Ilum