#pragma once

#include <Core/Time.hpp>

namespace Ilum
{
class Window;
class RHIContext;
class Editor;
class Renderer;
class Scene;
class ResourceManager;

class Engine
{
  public:
	Engine();

	~Engine();

	void Tick();

  private:
	std::unique_ptr<Window>     m_window      = nullptr;
	std::unique_ptr<RHIContext> m_rhi_context = nullptr;
	std::unique_ptr<Scene>      m_scene       = nullptr;
	std::unique_ptr<ResourceManager> m_resource_manager = nullptr;
	std::unique_ptr<Renderer> m_renderer = nullptr;
	std::unique_ptr<Editor> m_editor = nullptr;

	Timer m_timer;
};
}        // namespace Ilum