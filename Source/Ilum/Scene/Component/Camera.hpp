#pragma once

#include "Component.hpp"

#include <Geometry/Frustum.hpp>

#include <glm/glm.hpp>

#include <string>

namespace Ilum::cmpt
{
enum class CameraType : int32_t
{
	Perspective,
	Orthographic
};

struct Camera : public Component
{
	CameraType type = CameraType::Perspective;

	glm::mat4 view;
	glm::mat4 projection;
	glm::mat4 view_projection;

	float near_plane = 0.01f;
	float far_plane  = 1000.f;

	// Perspective
	float aspect = 1.f;
	float fov    = 45.f;

	// Orthographic
	float left   = -1.f;
	float right  = 1.f;
	float bottom = -1.f;
	float top    = 1.f;

	uint32_t frame_count = 0;

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(view, projection, view_projection, near_plane, far_plane,
		   aspect, fov, left, right, bottom, top);
	}

	glm::vec4 WorldToScreen(glm::vec3 position, glm::vec2 extent, glm::vec2 offset);
	glm::vec3 ScreenToWorld(glm::vec4 position, glm::vec2 extent, glm::vec2 offset);

	bool OnImGui(ImGuiContext &context) override;
};
}        // namespace Ilum::cmpt