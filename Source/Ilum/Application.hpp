#pragma once

#include <Core/Window.hpp>

#include <RHI/Device.hpp>

#include <Render/Renderer.hpp>

#include <Scene/Scene.hpp>

namespace Ilum
{
class Application
{
  public:
	Application();
	~Application() = default;

	void Tick();

  private:
	Window    m_window;
	RHIDevice m_device;
	ImGuiContext m_imgui_context;
	Renderer  m_renderer;
	std::unique_ptr<Scene> m_scene;
};
}        // namespace Ilum