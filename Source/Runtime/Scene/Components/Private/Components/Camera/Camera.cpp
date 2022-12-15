#include "Camera/Camera.hpp"

namespace Ilum
{
namespace Cmpt
{
Camera::Camera(const char *name, Node *node):
    Component(name, node)
{
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