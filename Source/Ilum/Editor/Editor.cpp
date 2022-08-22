#include "Editor.hpp"

#include "ImGui/ImGuiContext.hpp"

#include "Widget/RenderGraphEditor.hpp"

#include <imgui.h>

namespace Ilum
{
Editor::Editor(Window *window, RHIContext *rhi_context) :
    m_imgui_context(std::make_unique<ImGuiContext>(rhi_context, window))
{
	m_widgets.emplace_back(std::make_unique<RenderGraphEditor>(this));
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
			for (auto& widget : m_widgets)
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
}        // namespace Ilum