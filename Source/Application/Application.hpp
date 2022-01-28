#pragma once

#include <Core/Event/ApplicationEvent.hpp>
#include <Core/Event/KeyEvent.hpp>
#include <Core/Event/Event.hpp>
#include <Core/Timer.hpp>
#include <Core/Window.hpp>

namespace Ilum::App
{
class Application
{
  public:
	Application(Core::GraphicsBackend backend = Core::GraphicsBackend::Vulkan);

	~Application();

	void Run();

  private:
	void OnEvent(const Core::Event &event);

	bool OnWindowClosed(const Core::WindowClosedEvent &event);

	bool OnWindowResized(const Core::WindowResizedEvent &event);

  private:
	bool m_running   = true;
	bool m_minimized = false;

  private:
	std::shared_ptr<Core::Window> m_window = nullptr;

	Core::Timer timer;
};
}        // namespace Ilum::App