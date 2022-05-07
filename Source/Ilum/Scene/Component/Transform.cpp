#include "Transform.hpp"

#include <imgui.h>
#include <imgui_internal.h>

namespace Ilum::cmpt
{
inline bool DrawVec3Control(const std::string &label, glm::vec3 &values, float resetValue = 0.0f, float columnWidth = 100.0f)
{
	ImGuiIO &io        = ImGui::GetIO();
	auto     bold_font = io.Fonts->Fonts[0];
	bool     update    = false;

	ImGui::PushID(label.c_str());

	ImGui::Columns(2);
	ImGui::SetColumnWidth(0, columnWidth);
	ImGui::Text(label.c_str());
	ImGui::NextColumn();

	ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});

	float  line_height = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
	ImVec2 button_size = {line_height + 3.0f, line_height};

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
	ImGui::PushFont(bold_font);
	if (ImGui::Button("X", button_size))
	{
		values.x = resetValue;
		update   = true;
	}
	ImGui::PopFont();
	ImGui::PopStyleColor(3);

	ImGui::SameLine();
	update = ImGui::DragFloat("##X", &values.x, 0.1f, 0.0f, 0.0f, "%.3f") || update;
	ImGui::PopItemWidth();
	ImGui::SameLine();

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.2f, 0.7f, 0.2f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.3f, 0.8f, 0.3f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.2f, 0.7f, 0.2f, 1.0f});
	ImGui::PushFont(bold_font);
	if (ImGui::Button("Y", button_size))
	{
		values.y = resetValue;
		update   = true;
	}
	ImGui::PopFont();
	ImGui::PopStyleColor(3);

	ImGui::SameLine();
	update = ImGui::DragFloat("##Y", &values.y, 0.1f, 0.0f, 0.0f, "%.3f") || update;
	ImGui::PopItemWidth();
	ImGui::SameLine();

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.1f, 0.25f, 0.8f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.2f, 0.35f, 0.9f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.1f, 0.25f, 0.8f, 1.0f});
	ImGui::PushFont(bold_font);
	if (ImGui::Button("Z", button_size))
	{
		values.z = resetValue;
		update   = true;
	}
	ImGui::PopFont();
	ImGui::PopStyleColor(3);

	ImGui::SameLine();
	update = ImGui::DragFloat("##Z", &values.z, 0.1f, 0.0f, 0.0f, "%.3f") || update;
	ImGui::PopItemWidth();

	ImGui::PopStyleVar();

	ImGui::Columns(1);

	ImGui::PopID();

	return update;
}

bool Transform::OnImGui(ImGuiContext &context)
{
	bool update = false;
	update |= DrawVec3Control("Translation", translation, 0.f);
	update |= DrawVec3Control("Rotation", rotation, 0.f);
	update |= DrawVec3Control("Scale", scale, 1.f);
	return update;
}

}        // namespace Ilum::cmpt