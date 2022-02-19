#include "Editor.hpp"
#include "Panels/RenderGraphEditor.hpp"

#include <imgui.h>

namespace Ilum::Editor
{
Editor::Editor()
{
	m_panels.emplace_back(std::make_unique<RenderGraphEditor>());
}

void Editor::Show()
{
	ShowMenu();
	ShowPanels();
	ImGui::ShowDemoWindow();
}

void Editor::ShowPanels()
{
	for (auto& panel : m_panels)
	{
		panel->Show();
	}
}

void Editor::ShowMenu()
{
	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("Panels"))
		{
			for (auto &panel : m_panels)
			{
				ImGui::MenuItem(panel->GetName().c_str(), nullptr, &panel->active);
			}
			ImGui::EndMenu();
		}

		ImGui::EndMainMenuBar();
	}
}
}        // namespace Ilum::Editor