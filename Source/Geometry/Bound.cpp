#include "Bound.hpp"

#include <algorithm>

namespace Ilum::Geo
{
Bound::Bound(const glm::vec3 &_min, const glm::vec3 &_max) :
    m_min(_min), m_max(_max)
{
}

Bound::operator bool() const
{
	return m_min.x < m_max.x && m_min.y < m_max.y && m_min.z < m_max.z;
}

void Bound::Merge(const glm::vec3 &point)
{
	m_max = glm::max(m_max, point);
	m_min = glm::min(m_min, point);
}

void Bound::Merge(const std::vector<glm::vec3> &points)
{
	std::for_each(points.begin(), points.end(), [this](const glm::vec3 &p) { Merge(p); });
}

void Bound::Merge(const Bound &bounding_box)
{
	m_min.x = bounding_box.m_min.x < m_min.x ? bounding_box.m_min.x : m_min.x;
	m_min.y = bounding_box.m_min.y < m_min.y ? bounding_box.m_min.y : m_min.y;
	m_min.z = bounding_box.m_min.z < m_min.z ? bounding_box.m_min.z : m_min.z;
	m_max.x = bounding_box.m_max.x > m_max.x ? bounding_box.m_max.x : m_max.x;
	m_max.y = bounding_box.m_max.y > m_max.y ? bounding_box.m_max.y : m_max.y;
	m_max.z = bounding_box.m_max.z > m_max.z ? bounding_box.m_max.z : m_max.z;
}

Bound Bound::Transform(const glm::mat4 &trans) const
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

	return Bound(v[0], v[1]);
}

const glm::vec3 Bound::Center() const
{
	return (m_max + m_min) / 2.f;
}

const glm::vec3 Bound::Scale() const
{
	return m_max - m_min;
}

const glm::vec3 &Bound::GetMin() const
{
	return m_min;
}

const glm::vec3 &Bound::GetMax() const
{
	return m_max;
}

bool Bound::IsInside(const glm::vec3 &point) const
{
	return !(point.x < m_min.x ||
	         point.x > m_max.x ||
	         point.y < m_min.y ||
	         point.y > m_max.y ||
	         point.z < m_min.z ||
	         point.z > m_max.z);
}
}        // namespace Ilum::Geo