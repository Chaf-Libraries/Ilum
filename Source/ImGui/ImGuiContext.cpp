#include "ImGuiContext.hpp"

namespace Ilum
{
ImGuiContext::ImGuiContext();
ImGuiContext::~ImGuiContext();

void ImGuiContext::Initialize();
void ImGuiContext::Destroy();
void ImGuiContext::Recreate();
void ImGuiContext::BeginImGui();
void ImGuiContext::EndImGui();
void ImGuiContext::Render(const Graphics::CommandBuffer &cmd_buffer);
void ImGuiContext::Flush();
void ImGuiContext::SetStyle();
}