#pragma once

#include <entt.hpp>

#include "Scene/Entity.hpp"

#include <glm/glm.hpp>

namespace Ilum::cmpt
{
struct Camera
{
	inline static bool update = true;
};

struct PerspectiveCamera : public Camera
{
	float aspect     = 1.f;
	float fov        = 45.f;
	float far_plane  = 1000.f;
	float near_plane = 0.01f;
};

struct OrthographicCamera : public Camera
{
	float left       = -1.f;
	float right      = 1.f;
	float bottom     = -1.f;
	float top        = 1.f;
	float near_plane = 0.0f;
	float far_plane  = 1.f;
};
}        // namespace Ilum::cmpt