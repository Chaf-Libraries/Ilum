#include "Frustum.hpp"
#include "Bound.hpp"

namespace Ilum::Geo
{
Frustum::Frustum(const glm::mat4 &view_projection)
{
	glm::mat4 inverse = glm::inverse(view_projection);

	// Left
	{
		glm::vec3 normal = {
		    view_projection[0].w + view_projection[0].x,
		    view_projection[1].w + view_projection[1].x,
		    view_projection[2].w + view_projection[2].x};
		float constant = view_projection[3].w + view_projection[3].x;
		float length   = glm::length(normal);
		m_planes[0]    = Plane(normal / length, constant / length);
	}

	// Right
	{
		glm::vec3 normal = {
		    view_projection[0].w - view_projection[0].x,
		    view_projection[1].w - view_projection[1].x,
		    view_projection[2].w - view_projection[2].x};
		float constant = view_projection[3].w - view_projection[3].x;
		float length   = glm::length(normal);
		m_planes[1]    = Plane(normal / length, constant / length);
	}

	// Top
	{
		glm::vec3 normal = {
		    view_projection[0].w - view_projection[0].y,
		    view_projection[1].w - view_projection[1].y,
		    view_projection[2].w - view_projection[2].y};
		float constant = view_projection[3].w - view_projection[3].y;
		float length   = glm::length(normal);
		m_planes[2]    = Plane(normal / length, constant / length);
	}

	// Bottom
	{
		glm::vec3 normal = {
		    view_projection[0].w + view_projection[0].y,
		    view_projection[1].w + view_projection[1].y,
		    view_projection[2].w + view_projection[2].y};
		float constant = view_projection[3].w + view_projection[3].y;
		float length   = glm::length(normal);
		m_planes[3]    = Plane(normal / length, constant / length);
	}

	// Near
	{
		glm::vec3 normal = {
		    view_projection[0].w + view_projection[0].z,
		    view_projection[1].w + view_projection[1].z,
		    view_projection[2].w + view_projection[2].z};
		float constant = view_projection[3].w + view_projection[3].z;
		float length   = glm::length(normal);
		m_planes[4]    = Plane(normal / length, constant / length);
	}

	// Far
	{
		glm::vec3 normal = {
		    view_projection[0].w - view_projection[0].z,
		    view_projection[1].w - view_projection[1].z,
		    view_projection[2].w - view_projection[2].z};
		float constant = view_projection[3].w - view_projection[3].z;
		float length   = glm::length(normal);
		m_planes[5]    = Plane(normal / length, constant / length);
	}
}

bool Frustum::IsInside(const glm::vec3 &p)
{
	for (auto &plane : m_planes)
	{
		if (glm::dot(plane.GetNormal(), p) + plane.GetConstant() < 0.f)
		{
			return false;
		}
	}
	return true;
}

bool Frustum::IsInside(const Bound &bbox)
{
	for (auto &plane : m_planes)
	{
		glm::vec3 p;
		p.x = plane.GetNormal().x < 0.f ? bbox.GetMin().x : bbox.GetMax().x;
		p.y = plane.GetNormal().y < 0.f ? bbox.GetMin().y : bbox.GetMax().y;
		p.z = plane.GetNormal().z < 0.f ? bbox.GetMin().z : bbox.GetMax().z;

		if (glm::dot(plane.GetNormal(), p) + plane.GetConstant() < 0.f)
		{
			return false;
		}
	}
	return true;
}
}        // namespace Ilum::Geo