#include "Editor.hpp"

#include <imgui.h>

#include "Device/Window.hpp"

#include "ImGui/ImGuiContext.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/RenderPass/ImGuiPass.hpp"
#include "Renderer/Renderer.hpp"

#include "Panels/AssetBrowser.hpp"
#include "Panels/Hierarchy.hpp"
#include "Panels/Inspector.hpp"
#include "Panels/RenderGraphViewer.hpp"
#include "Panels/SceneView.hpp"

namespace Ilum
{
Editor::Editor(Context *context) :
    TSubsystem<Editor>(context)
{
}

bool Editor::onInitialize()
{
	if (!Renderer::instance()->hasImGui())
	{
		Renderer::instance()->setImGui(true);
		Renderer::instance()->rebuild();
	}

	ImGuiContext::initialize();

	m_panels.emplace_back(createScope<panel::RenderGraphViewer>());
	m_panels.emplace_back(createScope<panel::Inspector>());
	m_panels.emplace_back(createScope<panel::Hierarchy>());
	m_panels.emplace_back(createScope<panel::AssetBrowser>());
	m_panels.emplace_back(createScope<panel::SceneView>());

	return true;
}

void Editor::onPreTick()
{
	ImGuiContext::begin();
}

void Editor::onTick(float delta_time)
{
	if (!Renderer::instance()->hasImGui())
	{
		return;
	}

	static bool open = true;
	ImGui::ShowDemoWindow(&open);

	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("Panel"))
		{
			for (auto &panel : m_panels)
			{
				ImGui::MenuItem(panel->name().c_str(), nullptr, &panel->active);
			}
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}

	for (auto &panel : m_panels)
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

	m_panels.clear();
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