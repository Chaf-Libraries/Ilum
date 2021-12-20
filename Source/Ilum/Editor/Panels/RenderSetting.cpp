#include "RenderSetting.hpp"

#include "Renderer/Renderer.hpp"

#include <imgui.h>
#include <imgui_internal.h>

namespace Ilum::panel
{
template <typename Callback>
inline void draw_node(const std::string &name, Callback callback)
{
	const ImGuiTreeNodeFlags tree_node_flags          = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;
	ImVec2                   content_region_available = ImGui::GetContentRegionAvail();

	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});
	float line_height = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
	bool  open        = ImGui::TreeNodeEx(name.c_str(), tree_node_flags, name.c_str());
	ImGui::PopStyleVar();

	if (open)
	{
		callback();
		ImGui::TreePop();
	}
}

RenderSetting::RenderSetting()
{
	m_name = "Render Setting";
}

void RenderSetting::draw(float delta_time)
{
	ImGui::Begin("Render Setting", &active);

	draw_node("Culling", []() {
		ImGui::Checkbox("Frustum Culling", reinterpret_cast<bool *>(&Renderer::instance()->Culling.frustum_culling));
		ImGui::Checkbox("Back Face Cone Culling", reinterpret_cast<bool *>(&Renderer::instance()->Culling.backface_culling));
		ImGui::Checkbox("Hi-z Occlusion Culling", reinterpret_cast<bool *>(&Renderer::instance()->Culling.occulsion_culling));
	});

	draw_node("Color Correction", []() {
		ImGui::DragFloat("Exposure", &Renderer::instance()->Color_Correction.exposure, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
		ImGui::DragFloat("Gamma", &Renderer::instance()->Color_Correction.gamma, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
	});

	draw_node("Bloom", []() {
		ImGui::Checkbox("Enable", reinterpret_cast<bool *>(&Renderer::instance()->Bloom.enable));
		ImGui::DragFloat("Threshold", &Renderer::instance()->Bloom.threshold, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
		ImGui::DragFloat("Scale", &Renderer::instance()->Bloom.scale, 0.001f, 0.f, std::numeric_limits<float>::max(), "%.3f");
		ImGui::DragFloat("Strength", &Renderer::instance()->Bloom.strength, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
	});

	ImGui::End();
}
}        // namespace Ilum::panel