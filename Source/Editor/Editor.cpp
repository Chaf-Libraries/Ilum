#include "Editor.hpp"

#include "Graphics/ImGui/imgui.h"

#include "Device/Window.hpp"

#include <Graphics/ImGui/ImGuiContext.hpp>

namespace Ilum
{
Editor::Editor(Context *context):
    TSubsystem<Editor>(context)
{

}

bool Editor::onInitialize()
{
	ImGuiContext::initialize();
	return true;
}

void Editor::onPreTick()
{
	ImGuiContext::begin();
}

void Editor::onTick(float delta_time)
{

	ImGui::ShowDemoWindow();
}

void Editor::onPostTick()
{
	ImGuiContext::end();
}

void Editor::onShutdown()
{
	ImGuiContext::destroy();
}


}        // namespace Ilum