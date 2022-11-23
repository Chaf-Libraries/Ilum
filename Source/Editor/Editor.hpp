#pragma once

#include <memory>
#include <vector>

namespace Ilum
{
class Window;
class RHIContext;
class GuiContext;
class Widget;
class Renderer;
class Node;

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

	Window *GetWindow() const;

	void SelectNode(Node *node = nullptr);

	Node *GetSelectedNode() const;

  private:
	std::unique_ptr<GuiContext> m_imgui_context = nullptr;

	RHIContext *p_rhi_context = nullptr;
	Renderer   *p_renderer    = nullptr;

	std::vector<std::unique_ptr<Widget>> m_widgets;

	Node *m_select = nullptr;
};
}        // namespace Ilum