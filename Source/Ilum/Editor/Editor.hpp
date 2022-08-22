#pragma once

namespace Ilum
{
class Window;
class RHIContext;
class ImGuiContext;
class Widget;

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
	std::vector<std::unique_ptr<Widget>> m_widgets;
};
}        // namespace Ilum