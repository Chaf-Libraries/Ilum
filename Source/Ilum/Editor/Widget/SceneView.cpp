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


	ImGui::End();
}
}        // namespace Ilum