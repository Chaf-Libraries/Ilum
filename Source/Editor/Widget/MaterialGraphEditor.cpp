#pragma once

#include "MaterialGraphEditor.hpp"

#include <imnodes.h>

namespace Ilum
{
MaterialGraphEditor::MaterialGraphEditor(Editor *editor) :
    Widget("Material Graph Editor", editor)
{
}

MaterialGraphEditor::~MaterialGraphEditor()
{
}

void MaterialGraphEditor::Tick()
{
	static auto editor_context = ImNodes::EditorContextCreate();

	ImGui::Begin(m_name.c_str(), &m_active, ImGuiWindowFlags_MenuBar);

	ImNodes::EditorContextSet(editor_context);

	ImGui::Columns(2);
	ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.8f);

	ImNodes::BeginNodeEditor();

	ImNodes::EndNodeEditor();

	{
		ImGui::BeginChild("Render Graph Inspector", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

		ImGui::PushItemWidth(ImGui::GetColumnWidth(1) * 0.7f);

		ImGui::PopItemWidth();

		if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
		{
			ImGui::SetScrollHereY(1.0f);
		}

		ImGui::EndChild();
	}

	ImGui::End();
}
}        // namespace Ilum