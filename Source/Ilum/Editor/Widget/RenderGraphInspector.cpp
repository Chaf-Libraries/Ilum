#include "RenderGraphInspector.hpp"
#include "Editor/ImGui/ImGuiHelper.hpp"
#include "Editor/Editor.hpp"

#include <Renderer/Renderer.hpp>

#include <imgui.h>

namespace Ilum
{
RenderGraphInspector::RenderGraphInspector(Editor *editor):
    Widget("Render Graph Inspector", editor)
{
}

RenderGraphInspector::~RenderGraphInspector()
{
}

void RenderGraphInspector::Tick()
{
	ImGui::Begin(m_name.c_str());

	auto *renderer = p_editor->GetRenderer();
	auto *render_graph = renderer->GetRenderGraph();
	if (render_graph)
	{
		auto configs = render_graph->GetPassConfigs();
		for (auto &[name, config] : configs)
		{
			ImGui::PushID(config);
			if (ImGui::TreeNode(name.c_str()))
			{
				ImGui::EditVariant(*config);
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
	}

	ImGui::End();
}
}        // namespace Ilum