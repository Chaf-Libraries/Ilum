#include "RenderGraphEditor.hpp"

#include <Vulkan/ImGui/imnodes.h>

#include <Renderer/Renderer.hpp>

#include <imgui.h>

namespace Ilum::Editor
{
RenderGraphEditor::RenderGraphEditor()
{
	m_name = "Render Graph Editor";
}

void RenderGraphEditor::Show()
{
	auto &render_graph = Renderer::GetInstance().GetRenderGraph();

	ImGui::Begin(m_name.c_str(), &active);

	if (ImGui::Button("New"))
	{
	}
	ImGui::SameLine();
	if (ImGui::Button("Load"))
	{
	}
	ImGui::SameLine();
	if (ImGui::Button("Save"))
	{
	}

	 ImNodes::BeginNodeEditor();

	 // Draw node
	 for (auto& pass : render_graph.GetPassNodes())
	 {
		 if (pass.OnImNode())
		 {
			 pass.OnImGui();
		 }
	 }

	ImNodes::EndNodeEditor();

	ImGui::End();
}
}        // namespace Ilum::Editor