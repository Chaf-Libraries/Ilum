#include "SceneView.hpp"
#include "Editor.hpp"

#include <CodeGeneration/Meta/SceneMeta.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>
#include <Scene/Component/HierarchyComponent.hpp>
#include <Scene/Component/TagComponent.hpp>
#include <Scene/Component/TransformComponent.hpp>
#include <Scene/Scene.hpp>

#include <imgui.h>

namespace Ilum
{
SceneView::SceneView(Editor *editor) :
    Widget("Scene View", editor)
{
}

SceneView::~SceneView()
{
}

void SceneView::Tick()
{
	ImGui::Begin(m_name.c_str());

	auto *renderer = p_editor->GetRenderer();
	renderer->SetViewport(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y);
	auto *present_texture = renderer->GetPresentTexture();

	ImGui::Image(present_texture, ImGui::GetContentRegionAvail());

	if (ImGui::BeginDragDropTarget())
	{
		if (const auto *pay_load = ImGui::AcceptDragDropPayload("Scene"))
		{
			ASSERT(pay_load->DataSize == sizeof(std::string));
			std::string uuid = *static_cast<std::string *>(pay_load->Data);

			auto *meta  = p_editor->GetRenderer()->GetResourceManager()->GetScene(uuid);
			auto *scene = p_editor->GetRenderer()->GetScene();

			if (meta)
			{
				std::ifstream is("Asset/Meta/" + uuid + ".meta", std::ios::binary);
				InputArchive  archive(is);
				std::string   filename;
				archive(ResourceType::Scene, uuid, filename);
				entt::snapshot_loader{(*scene)()}
				    .entities(archive)
				    .component<
				        TagComponent,
				        TransformComponent,
				        HierarchyComponent>(archive);
			}
		}
	}

	ImGui::End();
}
}        // namespace Ilum