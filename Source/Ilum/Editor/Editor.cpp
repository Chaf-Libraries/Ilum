#include "Editor.hpp"

#include <imgui.h>

#include "Device/Window.hpp"

#include "ImGui/ImGuiContext.hpp"

#include "File/FileSystem.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/RenderPass/ImGuiPass.hpp"
#include "Renderer/Renderer.hpp"

#include "Panels/AssetBrowser.hpp"
#include "Panels/Console.hpp"
#include "Panels/Hierarchy.hpp"
#include "Panels/Inspector.hpp"
#include "Panels/ProfilerMonitor.hpp"
#include "Panels/RenderGraphViewer.hpp"
#include "Panels/RenderSetting.hpp"
#include "Panels/SceneView.hpp"

#include "Scene/Component/Light.hpp"
#include "Scene/Component/Renderable.hpp"
#include "Scene/Component/Camera.hpp"
#include "Scene/SceneSerializer.hpp"

#include "ImFileDialog.h"

namespace Ilum
{
Editor::Editor(Context *context) :
    TSubsystem<Editor>(context)
{
}

bool Editor::onInitialize()
{
	if (!Renderer::instance()->hasImGui())
	{
		Renderer::instance()->setImGui(true);
		Renderer::instance()->rebuild();
	}

	ImGuiContext::initialize();

	m_panels.emplace_back(createScope<panel::RenderGraphViewer>());
	m_panels.emplace_back(createScope<panel::Inspector>());
	m_panels.emplace_back(createScope<panel::Hierarchy>());
	m_panels.emplace_back(createScope<panel::AssetBrowser>());
	m_panels.emplace_back(createScope<panel::SceneView>());
	m_panels.emplace_back(createScope<panel::Console>());
	m_panels.emplace_back(createScope<panel::RenderSetting>());
	m_panels.emplace_back(createScope<panel::ProfilerMonitor>());

	return true;
}

void Editor::onPreTick()
{
	
}

void Editor::onTick(float delta_time)
{
	ImGuiContext::begin();

	if (!Renderer::instance()->hasImGui())
	{
		return;
	}

	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("New Scene"))
			{
				Scene::instance()->clear();
				Renderer::instance()->getResourceCache().clear();
				m_scene_path.clear();
				LOG_INFO("Create new scene");
			}

			if (ImGui::MenuItem("Open Scene"))
			{
				ifd::FileDialog::Instance().Open("OpenSceneDialog", "Open Scene", "Scene file (*.scene){.scene}");
			}

			if (ImGui::MenuItem("Save Scene"))
			{
				if (m_scene_path.empty())
				{
					ifd::FileDialog::Instance().Save("SaveSceneDialog", "Save Scene", "Scene file (*.scene){.scene}");
				}
				else
				{
					SceneSerializer serializer;
					serializer.serialize(m_scene_path.c_str());
					Scene::instance()->name = FileSystem::getFileName(m_scene_path, false);
					LOG_INFO("Save scene to {}", m_scene_path);
				}
			}

			if (ImGui::MenuItem("Save as ..."))
			{
				ifd::FileDialog::Instance().Save("SaveSceneDialog", "Save Scene", "Scene file (*.scene){.scene}");
			}

			ImGui::EndMenu();
		}

		if (ImGui::BeginMenu("Panel"))
		{
			for (auto &panel : m_panels)
			{
				ImGui::MenuItem(panel->name().c_str(), nullptr, &panel->active);
			}
			ImGui::EndMenu();
		}

		if (ImGui::BeginMenu("Entity"))
		{
			if (ImGui::MenuItem("Create Empty Entity"))
			{
				auto entity     = Scene::instance()->createEntity();
				m_select_entity = entity;
			}
			if (ImGui::BeginMenu("Light"))
			{
				if (ImGui::MenuItem("Directional Light"))
				{
					auto entity     = Scene::instance()->createEntity("Directional Light");
					m_select_entity = entity;

					entity.addComponent<cmpt::DirectionalLight>();
				}
				if (ImGui::MenuItem("Point Light"))
				{
					auto entity     = Scene::instance()->createEntity("Point Light");
					m_select_entity = entity;

					entity.addComponent<cmpt::PointLight>();
				}
				if (ImGui::MenuItem("Spot Light"))
				{
					auto entity     = Scene::instance()->createEntity("Spot Light");
					m_select_entity = entity;

					entity.addComponent<cmpt::SpotLight>();
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Camera"))
			{
				if (ImGui::MenuItem("Perspective Camera"))
				{
					auto entity     = Scene::instance()->createEntity("Perspective Camera");
					m_select_entity = entity;

					entity.addComponent<cmpt::PerspectiveCamera>();
				}
				if (ImGui::MenuItem("Orthographic Light"))
				{
					auto entity     = Scene::instance()->createEntity("Orthographic Light");
					m_select_entity = entity;

					entity.addComponent<cmpt::OrthographicCamera>();
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Shape"))
			{
				if (ImGui::MenuItem("Plane"))
				{
					auto entity                                     = Scene::instance()->createEntity("Plane");
					entity.addComponent<cmpt::MeshletRenderer>().model = std::string(PROJECT_SOURCE_DIR) + "Asset/Model/plane.obj";
					Renderer::instance()->getResourceCache().loadModelAsync(std::string(PROJECT_SOURCE_DIR) + "Asset/Model/plane.obj");
					m_select_entity = entity;
				}
				ImGui::EndMenu();
			}

			ImGui::EndMenu();
		}

		if (ifd::FileDialog::Instance().IsDone("OpenSceneDialog"))
		{
			if (ifd::FileDialog::Instance().HasResult())
			{
				m_scene_path = ifd::FileDialog::Instance().GetResult().u8string();
				SceneSerializer serializer;
				serializer.deserialize(m_scene_path.c_str());
			}
			ifd::FileDialog::Instance().Close();
		}

		if (ifd::FileDialog::Instance().IsDone("SaveSceneDialog"))
		{
			if (ifd::FileDialog::Instance().HasResult())
			{
				m_scene_path = ifd::FileDialog::Instance().GetResult().u8string();
				Scene::instance()->name = FileSystem::getFileName(m_scene_path, false);
				SceneSerializer serializer;
				serializer.serialize(m_scene_path.c_str());
				LOG_INFO("Save scene to {}", m_scene_path);
			}
			ifd::FileDialog::Instance().Close();
		}

		ImGui::EndMainMenuBar();
	}

	for (auto &panel : m_panels)
	{
		if (panel->active)
		{
			panel->draw(delta_time);
		}
	}

	ImGui::ShowDemoWindow();

	ImGuiContext::end();
}

void Editor::onPostTick()
{
	
}

void Editor::onShutdown()
{
	ImGuiContext::destroy();

	m_panels.clear();
}

void Editor::select(Entity entity)
{
	m_select_entity = entity;
}

Entity Editor::getSelect()
{
	return m_select_entity;
}

}        // namespace Ilum