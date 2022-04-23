#include "Frustum.hpp"

#include "BoundingBox.hpp"

namespace Ilum::geometry
{
Frustum::Frustum(const glm::mat4 &view_projection)
{
	// Left
	planes[0].normal.x = view_projection[0].w + view_projection[0].x;
	planes[0].normal.y = view_projection[1].w + view_projection[1].x;
	planes[0].normal.z = view_projection[2].w + view_projection[2].x;
	planes[0].constant = view_projection[3].w + view_projection[3].x;

	// Right
	planes[1].normal.x = view_projection[0].w - view_projection[0].x;
	planes[1].normal.y = view_projection[1].w - view_projection[1].x;
	planes[1].normal.z = view_projection[2].w - view_projection[2].x;
	planes[1].constant = view_projection[3].w - view_projection[3].x;

	// Top
	planes[2].normal.x = view_projection[0].w - view_projection[0].y;
	planes[2].normal.y = view_projection[1].w - view_projection[1].y;
	planes[2].normal.z = view_projection[2].w - view_projection[2].y;
	planes[2].constant = view_projection[3].w - view_projection[3].y;

	// Bottom
	planes[3].normal.x = view_projection[0].w + view_projection[0].y;
	planes[3].normal.y = view_projection[1].w + view_projection[1].y;
	planes[3].normal.z = view_projection[2].w + view_projection[2].y;
	planes[3].constant = view_projection[3].w + view_projection[3].y;

	// Near
	planes[4].normal.x = view_projection[0].w + view_projection[0].z;
	planes[4].normal.y = view_projection[1].w + view_projection[1].z;
	planes[4].normal.z = view_projection[2].w + view_projection[2].z;
	planes[4].constant = view_projection[3].w + view_projection[3].z;

	// Far
	planes[5].normal.x = view_projection[0].w - view_projection[0].z;
	planes[5].normal.y = view_projection[1].w - view_projection[1].z;
	planes[5].normal.z = view_projection[2].w - view_projection[2].z;
	planes[5].constant = view_projection[3].w - view_projection[3].z;

	for (auto& plane : planes)
	{
		float length = glm::length(plane.normal);
		plane.normal /= length;
		plane.constant /= length;
	}
}

bool Frustum::isInside(const glm::vec3 &p)
{
	for (auto& plane : planes)
	{
		if (glm::dot(plane.normal, p) + plane.constant < 0.f)
		{
			return false;
		}
	}
	return true;
}

bool Frustum::isInside(const BoundingBox &bbox)
{
	for (auto &plane : planes)
	{
		glm::vec3 p;
		p.x = plane.normal.x < 0.f ? bbox.min_.x : bbox.max_.x;
		p.y = plane.normal.y < 0.f ? bbox.min_.y : bbox.max_.y;
		p.z = plane.normal.z < 0.f ? bbox.min_.z : bbox.max_.z;

		if (glm::dot(plane.normal, p) + plane.constant < 0.f)
		{
			return false;
		}
	}
	return true;
}
}        // namespace Ilum::geometry