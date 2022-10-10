#include "Editor.hpp"

#include "ImGui/ImGuiContext.hpp"

#include "Widget/MaterialGraphEditor.hpp"
#include "Widget/RenderGraphEditor.hpp"
#include "Widget/RenderGraphInspector.hpp"
#include "Widget/ResourceBrowser.hpp"
#include "Widget/SceneHierarchy.hpp"
#include "Widget/SceneInspector.hpp"
#include "Widget/SceneView.hpp"

#include <imgui.h>

namespace Ilum
{
Editor::Editor(Window *window, RHIContext *rhi_context, Renderer *renderer) :
    m_imgui_context(std::make_unique<GuiContext>(rhi_context, window)), p_renderer(renderer), p_rhi_context(rhi_context)
{
	m_widgets.emplace_back(std::make_unique<ResourceBrowser>(this));
	m_widgets.emplace_back(std::make_unique<RenderGraphEditor>(this));
	m_widgets.emplace_back(std::make_unique<MaterialGraphEditor>(this));
	m_widgets.emplace_back(std::make_unique<RenderGraphInspector>(this));
	m_widgets.emplace_back(std::make_unique<SceneView>(this));
	m_widgets.emplace_back(std::make_unique<SceneInspector>(this));
	m_widgets.emplace_back(std::make_unique<SceneHierarchy>(this));
}

Editor::~Editor()
{
	m_imgui_context.reset();
}

void Editor::PreTick()
{
	m_imgui_context->BeginFrame();
}

void Editor::Tick()
{
	for (auto &widget : m_widgets)
	{
		if (widget->GetActive())
		{
			widget->Tick();
		}
	}

	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("Widget"))
		{
			for (auto &widget : m_widgets)
			{
				ImGui::MenuItem(widget->GetName().c_str(), nullptr, &widget->GetActive());
			}
			ImGui::EndMenu();
		}

		ImGui::EndMainMenuBar();
	}
}

void Editor::PostTick()
{
	m_imgui_context->EndFrame();
	m_imgui_context->Render();
}

Renderer *Editor::GetRenderer() const
{
	return p_renderer;
}

RHIContext *Editor::GetRHIContext() const
{
	return p_rhi_context;
}

void Editor::SelectEntity(const Entity &entity)
{
	m_select = entity;
}

Entity Editor::GetSelectedEntity() const
{
	return m_select;
}

}        // namespace Ilum