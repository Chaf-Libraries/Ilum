#pragma once

#include <Core/Core.hpp>

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

class EXPORT_API Editor
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
	struct Impl;
	Impl* m_impl = nullptr;
};
}        // namespace Ilum