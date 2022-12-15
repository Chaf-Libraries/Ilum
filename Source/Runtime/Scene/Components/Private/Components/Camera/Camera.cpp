#include "Camera/Camera.hpp"

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
	m_dirty |= (m_aspect != aspect);
	m_aspect = aspect;
}

float Camera::GetAspect()
{
	return m_aspect;
}

void Camera::SetNearPlane(float near_plane)
{
	m_dirty = (m_near != near_plane);
	m_near  = near_plane;
}

void Camera::SetFarPlane(float far_plane)
{
	m_dirty = (m_far != far_plane);
	m_far   = far_plane;
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
	Update();
	return m_view;
}

glm::mat4 Camera::GetProjectionMatrix()
{
	Update();
	return m_projection;
}

glm::mat4 Camera::GetViewProjectionMatrix()
{
	Update();
	return m_view_projection;
}

glm::mat4 Camera::GetInvViewMatrix()
{
	Update();
	return m_inv_view;
}

glm::mat4 Camera::GetInvProjectionMatrix()
{
	Update();
	return m_inv_projection;
}

glm::mat4 Camera::GetInvViewProjectionMatrix()
{
	Update();
	return m_inv_view_projection;
}
}        // namespace Cmpt
}        // namespace Ilum