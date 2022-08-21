#pragma once

namespace Ilum
{
class Window;
class RHIContext;
class ImGuiContext;

class Editor
{
  public:
	Editor(Window *window, RHIContext *rhi_context);

	~Editor();

	void PreTick();

	void Tick();

	void PostTick();

  private:
	std::unique_ptr<ImGuiContext> m_imgui_context = nullptr;
};
}        // namespace Ilum