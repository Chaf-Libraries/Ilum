#pragma once

#include <glm/glm.hpp>

#include "Geometry/Frustum.hpp"
#include "Geometry/Ray.hpp"

namespace Ilum
{
struct Camera
{
	enum class Type
	{
		Perspective,
		Orthographic
	}type = Type::Perspective;

	// Internal parameter
	float aspect = 1.f;
	float fov    = 30.f;
	float far_plane    = 500.f;
	float near_plane   = 0.01f;

	glm::vec3 forward = glm::normalize(glm::vec3(-10.f, -10.f, -10.f));
	glm::vec3 position  = {10.f, 10.f, 10.f};

	glm::mat4 view = {};
	glm::mat4 projection = {};
	glm::mat4 view_projection = {};

	bool update = true;

	Camera();

	~Camera() = default;

	void onUpdate();
};
}