#include "Camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include "Utils/PCH.hpp"

namespace Ilum
{
Camera::Camera()
{
	onUpdate();
}

void Camera::onUpdate()
{
	if (!update)
	{
		return;
	}

	auto right = glm::normalize(glm::cross(forward, glm::vec3{0.f, 1.f, 0.f}));
	auto up    = glm::normalize(glm::cross(right, forward));

	view = glm::lookAt(position, forward + position, up);

	projection = type == Type::Perspective ? glm::perspective(glm::radians(fov), aspect, near_plane, far_plane) :
                                             glm::ortho(-glm::radians(fov), glm::radians(fov), -glm::radians(fov) / aspect, glm::radians(fov) / aspect, near_plane, far_plane);

	view_projection = projection * view;

	frustum = geometry::Frustum(view_projection);

	update = false;
}
}        // namespace Ilum