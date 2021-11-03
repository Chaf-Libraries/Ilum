#pragma once

#include "Eventing/Event.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#undef far
#undef near

namespace Ilum::cmpt
{
enum class CameraType
{
	Perspective,
	Orthographic
};

struct Camera
{
	CameraType type = CameraType::Perspective;

	float aspect = 1.f;
	float fov    = 30.f;
	float far    = 10000.f;
	float near   = 0.01f;

	glm::mat4 view_projection;

	// TODO: Frustum
};
}        // namespace Ilum::Cmpt