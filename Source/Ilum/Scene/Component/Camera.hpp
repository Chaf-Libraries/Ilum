#pragma once

#include <entt.hpp>

#include "Scene/Entity.hpp"

#include "Geometry/Frustum.hpp"

#include <glm/glm.hpp>

namespace Ilum::cmpt
{
struct Camera
{
	geometry::Frustum frustum;

	glm::mat4 view;
	glm::mat4 projection;
	glm::mat4 view_projection;

	float near_plane = 0.01f;
	float far_plane  = 1000.f;

	glm::vec3 position;

	inline static bool update = true;
};

struct PerspectiveCamera : public Camera
{
	float aspect = 1.f;
	float fov    = 45.f;
};

struct OrthographicCamera : public Camera
{
	float left   = -1.f;
	float right  = 1.f;
	float bottom = -1.f;
	float top    = 1.f;
};
}        // namespace Ilum::cmpt