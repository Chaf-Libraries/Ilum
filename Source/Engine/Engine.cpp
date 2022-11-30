#include "Engine.hpp"
//#include "System.hpp"

#include <Core/Path.hpp>
#include <Core/Window.hpp>
#include <Editor/Editor.hpp>
#include <RHI/RHIContext.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>
#include <SceneGraph/Scene.hpp>

namespace Ilum
{
Engine::Engine()
{
	m_window           = std::make_unique<Window>("Ilum", "Asset/Icon/logo.bmp");
	m_rhi_context      = std::make_unique<RHIContext>(m_window.get(), "Vulkan", true);
	m_resource_manager = std::make_unique<ResourceManager>(m_rhi_context.get());
	m_scene            = std::make_unique<Scene>("Default Scene");
	m_renderer         = std::make_unique<Renderer>(m_rhi_context.get(), m_scene.get());
	m_editor           = std::make_unique<Editor>(m_window.get(), m_rhi_context.get(), m_renderer.get());

	Path::GetInstance().SetCurrent("./");

	auto resource = m_resource_manager->Import<ResourceType::Texture>("STB", "Asset/Texture/default.png");
}

Engine::~Engine()
{
	m_scene.reset();
	m_editor.reset();
	m_renderer.reset();
	m_resource_manager.reset();
	m_rhi_context.reset();
	m_window.reset();
}

void Engine::Tick()
{
	while (m_window->Tick())
	{
		m_timer.Tick();

		if (m_window->GetWidth() != 0 && m_window->GetHeight() != 0)
		{
			m_rhi_context->BeginFrame();

			// System::GetInstance().Tick(m_renderer.get());
			// m_scene->Tick();

			// Render loop
			m_renderer->Tick();

			// Render UI
			m_editor->PreTick();
			m_editor->Tick();
			m_editor->PostTick();

			m_rhi_context->EndFrame();
		}
		else
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(16));
		}

		m_window->SetTitle(fmt::format("IlumEngine | Scene - {} | Version - {} | GPU - {} | Backend - {} | CUDA - {} | FPS - {}",
		                               m_scene->GetName(),
		                               "1.0Beta",
		                               m_rhi_context->GetDeviceName(),
		                               m_rhi_context->GetBackend(),
		                               m_rhi_context->HasCUDA() ? "Enable" : "Disable",
		                               m_timer.FrameRate()));
	}
}
}        // namespace Ilum
