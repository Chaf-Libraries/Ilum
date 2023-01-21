#include "Engine.hpp"

#include <Core/Path.hpp>
#include <Core/Window.hpp>
#include <Editor/Editor.hpp>
#include <RHI/RHIContext.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>
#include <Scene/Scene.hpp>

namespace Ilum
{
Engine::Engine()
{
	m_window           = std::make_unique<Window>("Ilum", "Asset/Icon/logo.bmp");
	m_rhi_context      = std::make_unique<RHIContext>(m_window.get(), "Vulkan", false);
	m_resource_manager = std::make_unique<ResourceManager>(m_rhi_context.get());
	m_scene            = std::make_unique<Scene>("Default Scene");
	m_renderer         = std::make_unique<Renderer>(m_rhi_context.get(), m_scene.get(), m_resource_manager.get());
	m_editor           = std::make_unique<Editor>(m_window.get(), m_rhi_context.get(), m_renderer.get());

	Path::GetInstance().SetCurrent("./");

	//m_resource_manager->Import<ResourceType::Prefab>("./Asset/Model/SciFiHelmet/glTF/SciFiHelmet.gltf");
	//m_resource_manager->Import<ResourceType::Prefab>("./Asset/Model/vampire/dancing_vampire.dae");
	// m_resource_manager->Import<ResourceType::Prefab>("E:/Workspace/glTF-Sample-Models/2.0/Fox/glTF/Fox.gltf");

	//m_resource_manager->Import<ResourceType::Texture2D>("./Asset/Icon/Camera.png");
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

			// Update resource manager
			m_resource_manager->Tick();

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

		m_window->SetTitle(fmt::format("IlumEngine v{} | Scene - {} | FPS - {}",
		                               "1.0Beta",
		                               m_scene->GetName(),
		                               m_timer.FrameRate()));
	}
}
}        // namespace Ilum
