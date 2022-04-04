#pragma once

#include <entt.hpp>

#include "Scene/Entity.hpp"

#include "Geometry/Frustum.hpp"

#include <glm/glm.hpp>

#include <cereal/cereal.hpp>

namespace Ilum::cmpt
{
struct Camera
{
	geometry::Frustum frustum;

	glm::mat4 view;
	glm::mat4 projection;
	glm::mat4 view_projection;
	glm::mat4 last_view_projection;

	float near_plane = 0.01f;
	float far_plane  = 1000.f;

	glm::vec3 position;

	uint32_t frame_count = 0;

	inline static bool update = true;

	inline glm::vec4 world2Screen(glm::vec3 position, glm::vec2 extent, glm::vec2 offset)
	{
		glm::vec4 pos = view_projection * glm::vec4(position, 1.f);
		pos.x *= std::fabsf(0.5f / pos.w);
		pos.y *= std::fabsf(0.5f / pos.w);
		pos += glm::vec4(0.5f, 0.5f, 0.f, 0.f);
		pos.y = 1.f - pos.y;
		pos.x *= extent.x;
		pos.y *= extent.y;
		pos.x += offset.x;
		pos.y += offset.y;

		return pos;
	}

	inline glm::vec3 screen2World(glm::vec4 position, glm::vec2 extent, glm::vec2 offset)
	{
		glm::vec4 pos = position;
		pos.x -= offset.x;
		pos.y -= offset.y;
		pos.x /= extent.x;
		pos.y /= extent.y;
		pos.y = 1.f - pos.y;
		pos -= glm::vec4(0.5f, 0.5f, 0.f, 0.f);
		pos.x /= std::fabsf(0.5f / pos.w);
		pos.y /= std::fabsf(0.5f / pos.w);
		pos = glm::inverse(view_projection) * pos;

		return pos;
	}
};

struct PerspectiveCamera : public Camera
{
	float aspect = 1.f;
	float fov    = 45.f;

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(fov, aspect, near_plane, far_plane);
	}
};

struct OrthographicCamera : public Camera
{
	float left   = -1.f;
	float right  = 1.f;
	float bottom = -1.f;
	float top    = 1.f;

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(left, right, bottom, top, near_plane, far_plane);
	}
};
}        // namespace Ilum::cmpt