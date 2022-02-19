#include "Sphere.hpp"

#include <algorithm>

namespace Ilum::geometry
{
Sphere::Sphere(const glm::vec3 &center, float radius) :
    center(center), radius(radius)
{
}

void Sphere::merge(const glm::vec3 &point)
{
	glm::vec3 offset = center - point;
	float dist = glm::length(offset);

	if (dist > radius)
	{
		float half = (dist - radius) * 0.5f;
		radius += half;
		center += (half / dist) * offset;
	}
}

void Sphere::merge(const std::vector<glm::vec3> &points)
{
	std::for_each(points.begin(), points.end(), [this](const glm::vec3 &p) { merge(p); });
}

TriMesh Sphere::toTriMesh()
{


	return TriMesh();
}
}        // namespace Ilum::geometry