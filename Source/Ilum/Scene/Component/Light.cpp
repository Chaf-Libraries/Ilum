#include "Light.hpp"

#include <glm/gtc/type_ptr.hpp>

namespace Ilum::cmpt
{
bool Light::OnImGui(ImGuiContext &context)
{
	bool is_update = false;

	const char *const light_type[] = {"Point", "Directional", "Spot", "Area"};
	is_update |= ImGui::Combo("Type", reinterpret_cast<int32_t *>(&type), light_type, 4);

	is_update |= ImGui::ColorEdit3("Color", glm::value_ptr(color));
	is_update |= ImGui::DragFloat("Intensity", &intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
	if (type == LightType::Point)
	{
		is_update |= ImGui::DragFloat("Range", &range, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
	}
	else if (type == LightType::Spot)
	{
		is_update |= ImGui::DragFloat("Range", &range, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
		is_update |= ImGui::DragFloat("Cut off", &spot_inner_cone_angle, 0.0001f, 0.f, 1.f, "%.5f");
		is_update |= ImGui::DragFloat("Outer cut off", &spot_outer_cone_angle, 0.0001f, 0.f, spot_inner_cone_angle, "%.5f");
	}
	else if (type == LightType::Area)
	{
		const char *const light_shape[] = {"Rectangle", "Ellipse"};
		is_update |= ImGui::Combo("Shape", reinterpret_cast<int32_t *>(&shape), light_shape, 2);
	}

	if (type != LightType::Area)
	{
		if (ImGui::TreeNode("Shadow Attribute"))
		{
			switch (type)
			{
				case Ilum::cmpt::LightType::Point:
					ImGui::BulletText("OmniShadow - PCSS");
					break;
				case Ilum::cmpt::LightType::Directional:
					ImGui::BulletText("Cascade Shadow - PCF");
					break;
				case Ilum::cmpt::LightType::Spot:
					ImGui::BulletText("OmniShadow - PCSS");
					break;
				default:
					break;
			}
			is_update |= ImGui::DragInt("Filter Sample", &filter_sample, 0.1f, 0, std::numeric_limits<int32_t>::max());
			is_update |= ImGui::DragFloat("Filter Scale", &filter_scale, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
			ImGui::TreePop();
		}
	}

	return is_update;
}

}        // namespace Ilum::cmpt