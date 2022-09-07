#include "SceneView.hpp"
#include "Editor.hpp"

#include <Renderer/Renderer.hpp>

#include <imgui.h>

namespace Ilum
{
SceneView::SceneView(Editor *editor) :
    Widget("Scene View", editor)
{
}

SceneView::~SceneView()
{
}

void SceneView::Tick()
{
	ImGui::Begin(m_name.c_str());

	auto *renderer = p_editor->GetRenderer();
	renderer->SetViewport(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y);
	auto *present_texture = renderer->GetPresentTexture();
	if (present_texture)
	{
		ImGui::Image(present_texture, ImGui::GetContentRegionAvail());
	}

	ImGui::End();
}
}        // namespace Ilum