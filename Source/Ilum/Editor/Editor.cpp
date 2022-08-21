#include "Editor.hpp"
#include "ImGui/ImGuiContext.hpp"

#include <imgui.h>

namespace Ilum
{
Editor::Editor(Window *window, RHIContext *rhi_context) :
    m_imgui_context(std::make_unique<ImGuiContext>(rhi_context, window))
{
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
	ImGui::ShowDemoWindow();
}

void Editor::PostTick()
{
	m_imgui_context->EndFrame();
	m_imgui_context->Render();
}
}        // namespace Ilum