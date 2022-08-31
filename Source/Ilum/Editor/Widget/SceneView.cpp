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

	ImGui::Image(p_editor->GetRenderer()->GetTexture(), ImGui::GetContentRegionAvail());

	ImGui::End();
}
}        // namespace Ilum