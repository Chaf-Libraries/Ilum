#include "Application.hpp"

#include <Core/Logger.hpp>
#include <Core/Input.hpp>

#include <functional>
#include <thread>

namespace Ilum::App
{
Application::Application(Core::GraphicsBackend backend)
{
	Core::WindowDesc desc;
	desc.backend = backend;
	desc.title   = "IlumEngine";
	desc.vsync   = false;
	desc.width   = 1920;
	desc.height   = 1080;

	m_window = Core::Window::Create(desc);
	m_window->SetEventCallback(std::bind(&Application::OnEvent, this, std::placeholders::_1));
	Core::Window::SetInstance(m_window);
}

Application::~Application()
{
}

void Application::Run()
{
	auto title = m_window->GetTitle();
	while (m_running)
	{
		timer.OnUpdate();

		// Update input states
		Core::Input::Flush();
		
		m_window->OnUpdate();

		std::this_thread::sleep_for(std::chrono::milliseconds(10));

		if (Core::Input::IsKeyHeld(Core::Key::Escape))
		{
			m_running = false;
		}

		m_window->SetTitle(title + " FPS: " + std::to_string(timer.GetFPS()));
	}
}

void Application::OnEvent(const Core::Event &event)
{
	Core::EventDispatcher dispatcher(event);
	dispatcher.Dispatch<Core::WindowClosedEvent>(std::bind(&Application::OnWindowClosed, this, std::placeholders::_1));
	dispatcher.Dispatch<Core::WindowResizedEvent>(std::bind(&Application::OnWindowResized, this, std::placeholders::_1));

	Core::Input::OnEvent(event);
}

bool Application::OnWindowClosed(const Core::WindowClosedEvent &event)
{
	m_running = false;
	return true;
}

bool Application::OnWindowResized(const Core::WindowResizedEvent &event)
{
	if (event.GetWidth() == 0 || event.GetHeight() == 0)
	{
		m_minimized = true;
		return true;
	}

	m_minimized = false;

	return true;
}

}        // namespace Ilum::App