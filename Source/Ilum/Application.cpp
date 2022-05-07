#include "Application.hpp"

#include <Core/Time.hpp>

namespace Ilum
{
Application::Application() :
    m_window("Ilum", "Asset/Icon/logo.bmp", 1920, 1080),
    m_device(&m_window),
    m_imgui_context(&m_window, &m_device),
    m_renderer(&m_device)
{
	m_scene = std::make_unique<Scene>(&m_device, "Untitle Scene");
	m_renderer.SetScene(m_scene.get());
}

void Application::Tick()
{
	while (m_window.Tick())
	{
		Timer::GetInstance().Tick();

		m_device.NewFrame();
		{
			m_scene->Tick();

			m_imgui_context.BeginFrame();
			{
				m_renderer.OnImGui(m_imgui_context);
			}
			m_imgui_context.EndFrame();

			m_renderer.Tick();
			m_imgui_context.Render();
		}
		m_device.EndFrame();
	}
}

}        // namespace Ilum