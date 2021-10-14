#include "Editor.hpp"

#include "Graphics/ImGui/imgui.h"

#include "Device/Window.hpp"

namespace Ilum
{
Editor::Editor(Context *context):
    TSubsystem<Editor>(context)
{
}

void Editor::onPreTick()
{
	// Begin docking space
	//ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
	//window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
	//window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
	//ImGuiViewport *viewport = ImGui::GetMainViewport();
	//ImGui::SetNextWindowPos(viewport->WorkPos);
	//ImGui::SetNextWindowSize(viewport->WorkSize);
	//ImGui::SetNextWindowViewport(viewport->ID);
	//ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	//ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	//ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	//ImGui::Begin("DockSpace", (bool *) 1, window_flags);
	//ImGui::PopStyleVar();
	//ImGui::PopStyleVar(2);

	//ImGuiIO &io = ImGui::GetIO();
	//if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
	//{
	//	ImGuiID dockspace_id = ImGui::GetID("DockSpace");
	//	ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
	//}
}

void Editor::onTick(float delta_time)
{
	//ImGui::ShowDemoWindow();
}

void Editor::onPostTick()
{
	//// End docking space
	//ImGui::End();

	//ImGuiIO &io = ImGui::GetIO();
	//io.DisplaySize = ImVec2(static_cast<float>(Window::instance()->getWidth()), static_cast<float>(Window::instance()->getHeight()));
}
}        // namespace Ilum