#include "Camera.hpp"

#include <imgui.h>

namespace Ilum::cmpt
{
glm::vec4 Camera::WorldToScreen(glm::vec3 position, glm::vec2 extent, glm::vec2 offset)
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

glm::vec3 Camera::ScreenToWorld(glm::vec4 position, glm::vec2 extent, glm::vec2 offset)
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

bool Camera::OnImGui(ImGuiContext &context)
{
	bool is_update = false;

	const char *const camera_types[] = {"Perspective", "Orthographic"};
	is_update |= ImGui::Combo("Camera Type", reinterpret_cast<int32_t *>(&type), camera_types, 2);

	if (type == CameraType::Perspective)
	{
		is_update |= ImGui::DragFloat("Aspect", &aspect, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
		is_update |= ImGui::DragFloat("Fov", &fov, 0.01f, 0.f, 90.f, "%.3f");
	}
	else
	{
		is_update |= ImGui::DragFloat("Left", &left, 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f");
		is_update |= ImGui::DragFloat("Right", &right, 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f");
		is_update |= ImGui::DragFloat("Bottom", &bottom, 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f");
		is_update |= ImGui::DragFloat("Top", &top, 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f");
	}

	is_update |= ImGui::DragFloat("Near Plane", &near_plane, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
	is_update |= ImGui::DragFloat("Far Plane", &far_plane, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");

	return is_update;
}

}        // namespace Ilum::cmpt