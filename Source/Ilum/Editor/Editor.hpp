#pragma once

namespace Ilum
{
class Window;
class RHIContext;
class GuiContext;
class Widget;
class Renderer;

class Editor
{
  public:
	Editor(Window *window, RHIContext *rhi_context, Renderer *renderer);

	~Editor();

	void PreTick();

	void Tick();

	void PostTick();

	Renderer *GetRenderer() const;

	RHIContext *GetRHIContext() const;

  private:
	std::unique_ptr<GuiContext> m_imgui_context = nullptr;

	RHIContext *p_rhi_context = nullptr;
	Renderer *p_renderer = nullptr;

	std::vector<std::unique_ptr<Widget>> m_widgets;
};
}        // namespace Ilum