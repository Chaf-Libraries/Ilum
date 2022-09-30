#include "SceneHierarchy.hpp"
#include "Editor/Editor.hpp"

#include <Core/Path.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>
#include <Resource/ResourceMeta.hpp>
#include <Scene/Component/HierarchyComponent.hpp>
#include <Scene/Component/TagComponent.hpp>
#include <Scene/Component/TransformComponent.hpp>
#include <Scene/Entity.hpp>

#include <imgui.h>

#pragma warning(push, 0)
#include <nfd.h>
#pragma warning(pop)

namespace Ilum
{
inline void DeleteNode(Scene *scene, Entity &entity)
{
	if (!entity)
	{
		return;
	}

	auto child   = Entity(scene, entity.GetComponent<HierarchyComponent>().first);
	auto sibling = child ? Entity(scene, child.GetComponent<HierarchyComponent>().next) : Entity(nullptr, entt::null);
	while (sibling)
	{
		auto tmp = Entity(scene, sibling.GetComponent<HierarchyComponent>().next);
		DeleteNode(scene, sibling);
		sibling = tmp;
	}
	DeleteNode(scene, child);
	entity.Destroy();
}

inline bool IsParentOf(Scene *scene, Entity &lhs, Entity &rhs)
{
	auto parent = Entity(scene, rhs.GetComponent<HierarchyComponent>().parent);
	while (parent)
	{
		if (parent == lhs)
		{
			return true;
		}
		parent = Entity(scene, parent.GetComponent<HierarchyComponent>().parent);
	}
	return false;
}

inline void SetAsSon(Scene *scene, Entity &new_parent, Entity &new_son)
{
	if (new_parent && IsParentOf(scene, new_son, new_parent))
	{
		return;
	}

	auto &h2 = new_son.GetComponent<HierarchyComponent>();

	if (h2.next != entt::null)
	{
		Entity(scene, h2.next).GetComponent<HierarchyComponent>().prev = h2.prev;
	}

	if (h2.prev != entt::null)
	{
		Entity(scene, h2.prev).GetComponent<HierarchyComponent>().next = h2.next;
	}

	if (h2.parent != entt::null && new_son == Entity(scene, h2.parent).GetComponent<HierarchyComponent>().first)
	{
		Entity(scene, h2.parent).GetComponent<HierarchyComponent>().first = h2.next;
	}

	h2.next   = new_parent ? new_parent.GetComponent<HierarchyComponent>().first : entt::null;
	h2.prev   = entt::null;
	h2.parent = new_parent ? new_parent.GetHandle() : entt::null;

	if (new_parent && new_parent.GetComponent<HierarchyComponent>().first != entt::null)
	{
		Entity(scene, new_parent.GetComponent<HierarchyComponent>().first).GetComponent<HierarchyComponent>().prev = new_son.GetHandle();
	}

	if (new_parent)
	{
		new_parent.GetComponent<HierarchyComponent>().first = new_son.GetHandle();
	}
}

SceneHierarchy::SceneHierarchy(Editor *editor) :
    Widget("Scene Hierarchy", editor)
{
}

SceneHierarchy::~SceneHierarchy()
{
}

void SceneHierarchy::Tick()
{
	auto *scene = p_editor->GetRenderer()->GetScene();

	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Load Scene"))
			{
				char *path = nullptr;
				if (NFD_OpenDialog("scene", Path::GetInstance().GetCurrent(false).c_str(), &path) == NFD_OKAY)
				{
					std::string   dir      = Path::GetInstance().GetFileDirectory(path);
					std::string   filename = Path::GetInstance().GetFileName(path, false);
					std::ifstream is(dir + filename + ".scene", std::ios::binary);
					InputArchive  archive(is);
					entt::snapshot_loader{(*scene)()}
					    .entities(archive)
					    .component<
					        TagComponent,
					        TransformComponent,
					        HierarchyComponent>(archive);
					scene->SetName(filename);
					free(path);
				}
			}
			if (ImGui::MenuItem("Save Scene"))
			{
				char *path = nullptr;
				if (NFD_SaveDialog("scene", Path::GetInstance().GetCurrent(false).c_str(), &path) == NFD_OKAY)
				{
					std::string dir      = Path::GetInstance().GetFileDirectory(path);
					std::string filename = Path::GetInstance().GetFileName(path, false);
					// Save as .scene
					{
						std::ofstream os(dir + filename + ".scene", std::ios::binary);
						OutputArchive archive(os);
						entt::snapshot{(*scene)()}
						    .entities(archive)
						    .component<
						        TagComponent,
						        TransformComponent,
						        HierarchyComponent>(archive);
						scene->SetName(filename);
					}
					// Save as engine meta
					{
						std::string   uuid = std::to_string(Hash(filename));
						std::ofstream os("Asset/Meta/" + uuid + ".meta", std::ios::binary);
						OutputArchive archive(os);
						archive(ResourceType::Scene, uuid, filename);
						entt::snapshot{(*scene)()}
						    .entities(archive)
						    .component<
						        TagComponent,
						        TransformComponent,
						        HierarchyComponent>(archive);

						SceneMeta meta = {};
						meta.name      = filename;
						meta.uuid      = uuid;
						p_editor->GetRenderer()->GetResourceManager()->AddSceneMeta(std::move(meta));
					}
					free(path);
				}
			}
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}

	ImGui::Begin(m_name.c_str());

	if (!scene)
	{
		ImGui::End();
		return;
	}

	if (ImGui::BeginDragDropTarget())
	{
		if (const auto *pay_load = ImGui::AcceptDragDropPayload("Entity"))
		{
			ASSERT(pay_load->DataSize == sizeof(Entity));
			auto null_entity = Entity(nullptr, entt::null);
			SetAsSon(scene, null_entity, *static_cast<Entity *>(scene, pay_load->Data));
			static_cast<Entity *>(pay_load->Data)->GetComponent<TransformComponent>().update = true;
		}
	}

	scene->Execute([&](Entity &entity) {
		if (entity && entity.GetComponent<HierarchyComponent>().parent == entt::null)
		{
			DrawNode(entity);
		}
	});

	if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && ImGui::IsWindowHovered())
	{
		p_editor->SelectEntity();
	}

	if (ImGui::BeginPopupContextWindow(0, 1, false))
	{
		if (ImGui::MenuItem("New Entity"))
		{
			scene->CreateEntity("Untitled Entity");
		}
		ImGui::EndPopup();
	}

	ImGui::End();
}

void SceneHierarchy::DrawNode(Entity &entity)
{
	auto *scene = p_editor->GetRenderer()->GetScene();
	if (!entity || !scene)
	{
		return;
	}

	auto &tag = entity.GetComponent<TagComponent>().tag;

	bool has_child = entity.HasComponent<HierarchyComponent>() && entity.GetComponent<HierarchyComponent>().first != entt::null;

	// Setting up
	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));
	ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_DefaultOpen | (p_editor->GetSelectedEntity() == entity ? ImGuiTreeNodeFlags_Selected : 0) | (has_child ? 0 : ImGuiTreeNodeFlags_Leaf);

	ImGui::PushID((uint32_t) entity.GetHandle());
	bool open = ImGui::TreeNodeEx(std::to_string(entity).c_str(), flags, "%s", tag.c_str());

	ImGui::PopStyleVar();

	// Delete entity
	bool entity_deleted = false;
	if (ImGui::BeginPopupContextItem(std::to_string(entity).c_str()))
	{
		if (ImGui::MenuItem("Delete Entity"))
		{
			entity_deleted = true;
		}
		ImGui::EndPopup();
	}
	ImGui::PopID();

	// Select entity
	if (ImGui::IsItemClicked())
	{
		p_editor->SelectEntity(entity);
	}

	// Drag and drop
	if (ImGui::BeginDragDropSource())
	{
		if (entity.HasComponent<HierarchyComponent>())
		{
			ImGui::SetDragDropPayload("Entity", &entity, sizeof(Entity));
		}
		ImGui::EndDragDropSource();
	}

	if (ImGui::BeginDragDropTarget())
	{
		if (const auto *pay_load = ImGui::AcceptDragDropPayload("Entity"))
		{
			if (pay_load->DataSize == sizeof(Entity))
			{
				SetAsSon(scene, entity, *static_cast<Entity *>(pay_load->Data));
				entity.GetComponent<HierarchyComponent>().update                                 = true;
				static_cast<Entity *>(pay_load->Data)->GetComponent<HierarchyComponent>().update = true;
			}
		}
		ImGui::EndDragDropTarget();
	}

	// Recursively open children entities
	if (open)
	{
		if (has_child)
		{
			auto child = Entity(scene, entity.GetComponent<HierarchyComponent>().first);
			while (child)
			{
				DrawNode(child);
				if (child)
				{
					child = Entity(scene, child.GetComponent<HierarchyComponent>().next);
				}
			}
		}
		ImGui::TreePop();
	}

	// Delete callback
	if (entity_deleted)
	{
		auto prev   = Entity(scene, entity.GetComponent<HierarchyComponent>().prev);
		auto next   = Entity(scene, entity.GetComponent<HierarchyComponent>().next);
		auto parent = Entity(scene, entity.GetComponent<HierarchyComponent>().parent);
		auto first  = Entity(scene, entity.GetComponent<HierarchyComponent>().first);

		if (prev)
		{
			prev.GetComponent<HierarchyComponent>().next = entity.GetComponent<HierarchyComponent>().next;
		}

		if (next)
		{
			next.GetComponent<HierarchyComponent>().prev = entity.GetComponent<HierarchyComponent>().prev;
		}

		if (parent && entity == parent.GetComponent<HierarchyComponent>().first)
		{
			parent.GetComponent<HierarchyComponent>().first = entity.GetComponent<HierarchyComponent>().next;
		}

		DeleteNode(scene, entity);

		if (p_editor->GetSelectedEntity() == entity)
		{
			p_editor->SelectEntity();
		}
	}
}
}        // namespace Ilum