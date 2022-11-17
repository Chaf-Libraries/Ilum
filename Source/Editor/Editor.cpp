#include "Editor.hpp"
#include "Widget.hpp"

#include "ImGui/ImGuiContext.hpp"

#include <Core/Plugin.hpp>

#include <imgui.h>

#include <filesystem>
#include <regex>

namespace Ilum
{
Editor::Editor(Window *window, RHIContext *rhi_context, Renderer *renderer) :
    m_imgui_context(std::make_unique<GuiContext>(rhi_context, window)), p_renderer(renderer), p_rhi_context(rhi_context)
{
	for (const auto& file : std::filesystem::directory_iterator("./lib/"))
	{
		std::string filename = file.path().filename().string();
		if (std::regex_match(filename, std::regex("(Editor.)(.*)(.dll)")))
		{
			m_widgets.emplace_back(std::unique_ptr<Widget>(std::move(PluginManager::GetInstance().Call<Widget *>(filename, "Create", this, ImGui::GetCurrentContext()))));
		}
	}
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

Window *Editor::GetWindow() const
{
	return m_imgui_context->GetWindow();
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