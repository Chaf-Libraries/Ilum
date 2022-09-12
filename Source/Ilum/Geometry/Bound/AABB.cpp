#include "AABB.hpp"

namespace Ilum
{
AABB::AABB(const glm::vec3 &min, const glm::vec3 &max) :
    min(min), max(max)
{
}

void AABB::Merge(const glm::vec3 &point)
{
	min = glm::min(min, point);
	max = glm::max(max, point);
}

void AABB::Merge(const std::vector<glm::vec3> &points)
{
	for (auto &point : points)
	{
		Merge(point);
	}
}

void AABB::Merge(const AABB &aabb)
{
	min = glm::min(min, aabb.min);
	max = glm::max(max, aabb.max);
}

AABB AABB::Transform(const glm::mat4 &transform)
{
	glm::vec3 v[2] = {}, xa, xb, ya, yb, za, zb;

	xa = transform[0] * min[0];
	xb = transform[0] * max[0];

	ya = transform[1] * min[1];
	yb = transform[1] * max[1];

	za = transform[2] * min[2];
	zb = transform[2] * max[2];

	v[0] = transform[3];
	v[0] += glm::min(xa, xb);
	v[0] += glm::min(ya, yb);
	v[0] += glm::min(za, zb);

	v[1] = transform[3];
	v[1] += glm::max(xa, xb);
	v[1] += glm::max(ya, yb);
	v[1] += glm::max(za, zb);

	return AABB(v[0], v[1]);
}

const glm::vec3 AABB::Center() const
{
	return 0.5f * (min + max);
}

const glm::vec3 AABB::Scale() const
{
	return max - min;
}
}        // namespace Ilum