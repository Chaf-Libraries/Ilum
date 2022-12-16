#include "Camera/Camera.hpp"
#include "Transform.hpp"

#include <SceneGraph/Node.hpp>

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