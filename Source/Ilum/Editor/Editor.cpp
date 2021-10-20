#include "Editor.hpp"

#include <imgui.h>

#include "Device/Window.hpp"

#include "ImGui/ImGuiContext.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/RenderPass/ImGuiPass.hpp"
#include "Renderer/Renderer.hpp"

#include "Panels/RenderGraphViewer.hpp"
#include "Panels/Inspector.hpp"


namespace Ilum
{
Editor::Editor(Context *context) :
    TSubsystem<Editor>(context)
{

}

bool Editor::onInitialize()
{
	auto &rg = Renderer::instance()->getRenderGraph();
	if (!rg.hasRenderPass<ImGuiPass>())
	{
		auto output = Renderer::instance()->getRenderGraph().output();

		auto current_build                     = Renderer::instance()->buildRenderGraph;
		Renderer::instance()->buildRenderGraph = [current_build, output, &rg](RenderGraphBuilder &builder) {
			current_build(builder);
			builder.addRenderPass("ImGuiPass", createScope<ImGuiPass>(output, rg.empty() ? AttachmentState::Clear_Color : AttachmentState::Load_Color)).setOutput(output);
		};

		Renderer::instance()->rebuild();
	}

	ImGuiContext::initialize();

	m_panels.emplace_back(createScope<panel::RenderGraphViewer>());
	m_panels.emplace_back(createScope<panel::Inspector>());

	return true;
}

void Editor::onPreTick()
{
	ImGuiContext::begin();
}

void Editor::onTick(float delta_time)
{
	static bool open = true;
	ImGui::ShowDemoWindow(&open);

	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("Panel"))
		{
			for (auto& panel : m_panels)
			{
				ImGui::MenuItem(panel->name().c_str(), nullptr,&panel->active);
			}
			ImGui::EndMenu();
		}


		ImGui::EndMainMenuBar();
	}

	for (auto& panel : m_panels)
	{
		if (panel->active)
		{
			panel->draw();
		}
	}
}

void Editor::onPostTick()
{
	ImGuiContext::end();
}

void Editor::onShutdown()
{
	ImGuiContext::destroy();

}

void Editor::select(Entity entity)
{
	m_select_entity = entity;
}

Entity Editor::getSelect()
{
	return m_select_entity;
}

}        // namespace Ilum