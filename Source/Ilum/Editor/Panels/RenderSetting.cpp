#include "RenderSetting.hpp"

#include "Renderer/Renderer.hpp"

#include <imgui.h>
#include <imgui_internal.h>

namespace Ilum::panel
{
template <typename Callback>
inline void drawNode(const std::string &name, Callback callback)
{
	const ImGuiTreeNodeFlags tree_node_flags          = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;
	ImVec2                   content_region_available = ImGui::GetContentRegionAvail();

	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});
	float line_height = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
	bool open = ImGui::TreeNodeEx(name.c_str(), tree_node_flags, name.c_str());
	ImGui::PopStyleVar();

	if (open)
	{
		callback();
		ImGui::TreePop();
	}
}

void RenderSetting::draw(float delta_time)
{
	ImGui::Begin("Render Setting", &active);

	drawNode("Color Correction", []() {
		ImGui::DragFloat("Exposure", &Renderer::instance()->Color_Correction.exposure, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
		ImGui::DragFloat("Gamma", &Renderer::instance()->Color_Correction.gamma, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
	});

	drawNode("Bloom", []() {
		ImGui::Checkbox("Enable", reinterpret_cast<bool*>(&Renderer::instance()->Bloom.enable));
		ImGui::DragFloat("Threshold", &Renderer::instance()->Bloom.threshold, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
		ImGui::DragFloat("Scale", &Renderer::instance()->Bloom.scale, 0.001f, 0.f, std::numeric_limits<float>::max(), "%.3f");
		ImGui::DragFloat("Strength", &Renderer::instance()->Bloom.strength, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
	});

	ImGui::End();
}
}        // namespace Ilum::panel