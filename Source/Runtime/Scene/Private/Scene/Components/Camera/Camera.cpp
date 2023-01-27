#include "Components/Camera/Camera.hpp"
#include "Components/Transform.hpp"

#include <Scene/Node.hpp>

namespace Ilum
{
namespace Cmpt
{
Camera::Camera(const char *name, Node *node) :
    Component(name, node)
{
}

void Camera::SetAspect(float aspect)
{
	m_update |= (m_aspect != aspect);
	m_aspect = aspect;
}

float Camera::GetAspect()
{
	return m_aspect;
}

void Camera::SetNearPlane(float near_plane)
{
	m_update |= (near_plane != m_near);
	m_near   = near_plane;
}

void Camera::SetFarPlane(float far_plane)
{
	m_update |= (far_plane != m_far);
	m_far    = far_plane;
}

float Camera::GetNearPlane() const
{
	return m_near;
}

float Camera::GetFarPlane() const
{
	return m_far;
}

glm::mat4 Camera::GetViewMatrix()
{
	UpdateView();
	return m_view;
}

glm::mat4 Camera::GetProjectionMatrix()
{
	UpdateProjection();
	return m_projection;
}

glm::mat4 Camera::GetViewProjectionMatrix()
{
	return GetProjectionMatrix() * GetViewMatrix();
}

glm::mat4 Camera::GetInvViewMatrix()
{
	UpdateView();
	return m_inv_view;
}

glm::mat4 Camera::GetInvProjectionMatrix()
{
	UpdateProjection();
	return m_inv_projection;
}

glm::mat4 Camera::GetInvViewProjectionMatrix()
{
	return GetInvViewMatrix() * GetInvProjectionMatrix();
}

const std::array<glm::vec4, 6> &Camera::GetFrustumPlanes()
{
	glm::mat4 view_projection = GetViewProjectionMatrix();

	// Left
	m_frustum_planes[0].x = view_projection[0].w + view_projection[0].x;
	m_frustum_planes[0].y = view_projection[1].w + view_projection[1].x;
	m_frustum_planes[0].z = view_projection[2].w + view_projection[2].x;
	m_frustum_planes[0].w = view_projection[3].w + view_projection[3].x;

	// Right
	m_frustum_planes[1].x = view_projection[0].w - view_projection[0].x;
	m_frustum_planes[1].y = view_projection[1].w - view_projection[1].x;
	m_frustum_planes[1].z = view_projection[2].w - view_projection[2].x;
	m_frustum_planes[1].w = view_projection[3].w - view_projection[3].x;

	// Top
	m_frustum_planes[2].x = view_projection[0].w - view_projection[0].y;
	m_frustum_planes[2].y = view_projection[1].w - view_projection[1].y;
	m_frustum_planes[2].z = view_projection[2].w - view_projection[2].y;
	m_frustum_planes[2].w = view_projection[3].w - view_projection[3].y;

	// Bottom
	m_frustum_planes[3].x = view_projection[0].w + view_projection[0].y;
	m_frustum_planes[3].y = view_projection[1].w + view_projection[1].y;
	m_frustum_planes[3].z = view_projection[2].w + view_projection[2].y;
	m_frustum_planes[3].w = view_projection[3].w + view_projection[3].y;

	// Near
	m_frustum_planes[4].x = view_projection[0].w + view_projection[0].z;
	m_frustum_planes[4].y = view_projection[1].w + view_projection[1].z;
	m_frustum_planes[4].z = view_projection[2].w + view_projection[2].z;
	m_frustum_planes[4].w = view_projection[3].w + view_projection[3].z;

	// Far
	m_frustum_planes[5].x = view_projection[0].w - view_projection[0].z;
	m_frustum_planes[5].y = view_projection[1].w - view_projection[1].z;
	m_frustum_planes[5].z = view_projection[2].w - view_projection[2].z;
	m_frustum_planes[5].w = view_projection[3].w - view_projection[3].z;

	for (auto &plane : m_frustum_planes)
	{
		float length = glm::length(glm::vec3(plane.x, plane.y, plane.z));
		plane /= length;
	}

	return m_frustum_planes;
}

void Camera::UpdateView()
{
	if (m_inv_view != p_node->GetComponent<Cmpt::Transform>()->GetWorldTransform())
	{
		m_inv_view = p_node->GetComponent<Cmpt::Transform>()->GetWorldTransform();
		m_view     = glm::inverse(m_inv_view);
	}
}
}        // namespace Cmpt
}        // namespace Ilum