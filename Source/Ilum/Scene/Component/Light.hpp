#pragma once

#include "Component.hpp"

#include <RHI/Buffer.hpp>

#include <entt.hpp>

#include <imgui.h>
#include <imgui_internal.h>

namespace Ilum::cmpt
{
enum class LightType
{
	Point,
	Directional,
	Spot,
	Area
};

enum class AreaLightShape
{
	Rectangle,
	Ellipse
};

struct Light : public Component
{
	LightType type = LightType::Point;

	glm::vec3 color     = {1.f, 1.f, 1.f};
	float     intensity = 1.f;

	float range                 = 1.f;
	float spot_inner_cone_angle = glm::cos(glm::radians(12.5f));
	float spot_outer_cone_angle = glm::cos(glm::radians(17.5f));

	// Shadow Attribute
	float   filter_scale  = 3.f;
	int32_t filter_sample = 20;

	AreaLightShape shape = AreaLightShape::Rectangle;

	// Shadow map
	std::unique_ptr<Texture> shadow_map = nullptr;
	std::unique_ptr<Buffer>  buffer     = nullptr;

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(type, color, intensity, range, spot_inner_cone_angle, spot_outer_cone_angle, filter_scale, filter_sample);
	}

	bool OnImGui(ImGuiContext &context);
};
}        // namespace Ilum::cmpt