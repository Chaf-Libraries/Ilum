#include "Application.hpp"

#include <Core/Input.hpp>
#include <Core/Path.hpp>
#include <Core/Time.hpp>

#include <imgui.h>

namespace Ilum
{
Application::Application() :
    m_window("Ilum", "Asset/Icon/logo.bmp"),
    m_device(&m_window),
    m_imgui_context(&m_window, &m_device),
    m_renderer(std::make_unique<Renderer>(&m_device)),
    m_asset_manager(std::make_unique<AssetManager>(&m_device))
{
	m_scene = std::make_unique<Scene>(&m_device, *m_asset_manager, "Untitle Scene");
	m_renderer->SetScene(m_scene.get());
	Input::GetInstance().Bind(&m_window);
}

Application::~Application()
{
	m_scene.reset();
	m_asset_manager.reset();
	m_renderer.reset();
}

void Application::Tick()
{
	while (m_window.Tick())
	{
		Timer::GetInstance().Tick();

		if (m_window.m_height != 0 && m_window.m_width != 0)
		{
			m_device.NewFrame();

			m_imgui_context.BeginFrame();
			{
				if (ImGui::BeginMainMenuBar())
				{
					if (ImGui::BeginMenu("File"))
					{
						if (ImGui::MenuItem("New Scene"))
						{
							m_renderer.reset();
							m_asset_manager.reset();
							m_scene.reset();
							m_renderer      = std::make_unique<Renderer>(&m_device);
							m_asset_manager = std::make_unique<AssetManager>(&m_device);
							m_scene         = std::make_unique<Scene>(&m_device, *m_asset_manager, "Untitle Scene");
							m_renderer->SetScene(m_scene.get());
						}
						if (ImGui::MenuItem("Load Scene"))
						{
							m_imgui_context.OpenFileDialog("Load Scene", "Load Scene", "Scene file (*.scene){.scene}");
						}
						if (ImGui::MenuItem("Save Scene"))
						{
							if (!m_scene->GetSavePath().empty())
							{
								m_scene->Save();
							}
							else
							{
								m_imgui_context.OpenFileDialog("Save Scene", "Load Scene", "Scene file (*.scene){.scene}", false);
							}
						}
						if (ImGui::MenuItem("Save As..."))
						{
							m_imgui_context.OpenFileDialog("Load Scene As", "Load Scene As", "Scene file (*.scene){.scene}", false);
						}
						if (ImGui::MenuItem("Import gltf"))
						{
							m_imgui_context.OpenFileDialog("Import GLTF", "Import GLTF", "GLTF file (*.gltf;*.glb){.gltf,.glb}");
						}
						if (ImGui::MenuItem("Export gltf"))
						{
							m_imgui_context.OpenFileDialog("Export GLTF", "Export GLTF", "GLTF file (*.gltf;*.glb){.gltf,.glb}", false);
						}
						ImGui::EndMenu();
					}
					ImGui::EndMainMenuBar();
				}
				m_imgui_context.GetFileDialogResult("Load Scene", [this](const std::string &path) { m_scene->Load(path); });
				m_imgui_context.GetFileDialogResult("Save Scene", [this](const std::string &path) { m_scene->Save(path); });
				m_imgui_context.GetFileDialogResult("Load Scene As", [this](const std::string &path) { m_scene->Save(path); });
				m_imgui_context.GetFileDialogResult("Import GLTF", [this](const std::string &path) { m_scene->ImportGLTF(path); });
				m_imgui_context.GetFileDialogResult("Export GLTF", [this](const std::string &path) { m_scene->ExportGLTF(path); });

				m_renderer->OnImGui(m_imgui_context);
				m_asset_manager->OnImGui(m_imgui_context);
			}
			m_imgui_context.EndFrame();

			m_scene->Tick();
			m_renderer->Tick();

			m_imgui_context.Render();

			m_device.EndFrame();
		}
	}
}

}        // namespace Ilum