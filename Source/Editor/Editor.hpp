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

namespace Cmpt
{
class Camera;
}

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

	void SetMainCamera(Cmpt::Camera *camera = nullptr);

	Node *GetSelectedNode() const;

	Cmpt::Camera *GetMainCamera() const;

  private:
	struct Impl;
	Impl* m_impl = nullptr;
};
}        // namespace Ilum