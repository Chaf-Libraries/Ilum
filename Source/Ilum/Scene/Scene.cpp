#include "Scene.hpp"
#include "Entity.hpp"

#include "Component/Hierarchy.hpp"
#include "Component/Tag.hpp"
#include "Component/Transform.hpp"

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>

#include <imgui.h>

#include <fstream>

namespace Ilum
{
template <typename T>
inline std::string GetComponentName()
{
	std::string name   = typeid(T).name();
	size_t      offset = name.find_last_of(':');
	return name.substr(offset + 1, name.length());
}

// Draw Scene Hierarchy
inline void DeleteNode(Scene &scene, Entity entity)
{
	if (!entity)
	{
		return;
	}

	auto child   = Entity(scene, entity.GetComponent<cmpt::Hierarchy>().first);
	auto sibling = child ? Entity(scene, child.GetComponent<cmpt::Hierarchy>().next) : Entity(scene);
	while (sibling)
	{
		auto tmp = Entity(scene, sibling.GetComponent<cmpt::Hierarchy>().next);
		DeleteNode(scene, sibling);
		sibling = tmp;
	}
	DeleteNode(scene, child);
	entity.Destroy();
}

inline bool IsParentOf(Scene &scene, Entity lhs, Entity rhs)
{
	auto parent = Entity(scene, rhs.GetComponent<cmpt::Hierarchy>().parent);
	while (parent)
	{
		if (parent == lhs)
		{
			return true;
		}
		parent = Entity(scene, parent.GetComponent<cmpt::Hierarchy>().parent);
	}
	return false;
}

inline void SetAsSon(Scene &scene, Entity new_parent, Entity new_son)
{
	if (new_parent && IsParentOf(scene, new_son, new_parent))
	{
		return;
	}

	auto &h2 = new_son.GetComponent<cmpt::Hierarchy>();

	if (h2.next != entt::null)
	{
		Entity(scene, h2.next).GetComponent<cmpt::Hierarchy>().prev = h2.prev;
	}

	if (h2.prev != entt::null)
	{
		Entity(scene, h2.prev).GetComponent<cmpt::Hierarchy>().next = h2.next;
	}

	if (h2.parent != entt::null && new_son == Entity(scene, h2.parent).GetComponent<cmpt::Hierarchy>().first)
	{
		Entity(scene, h2.parent).GetComponent<cmpt::Hierarchy>().first = h2.next;
	}

	h2.next   = new_parent ? new_parent.GetComponent<cmpt::Hierarchy>().first : entt::null;
	h2.prev   = entt::null;
	h2.parent = new_parent ? new_parent.GetHandle() : entt::null;

	if (new_parent && new_parent.GetComponent<cmpt::Hierarchy>().first != entt::null)
	{
		Entity(scene, new_parent.GetComponent<cmpt::Hierarchy>().first).GetComponent<cmpt::Hierarchy>().prev = new_son;
	}

	if (new_parent)
	{
		new_parent.GetComponent<cmpt::Hierarchy>().first = new_son;
	}
}

inline void DrawNode(Scene &scene, Entity entity, Entity &select)
{
	if (!entity)
	{
		return;
	}

	auto &tag = entity.GetComponent<cmpt::Tag>().name;

	bool has_child = entity.HasComponent<cmpt::Hierarchy>() && entity.GetComponent<cmpt::Hierarchy>().first != entt::null;

	// Setting up
	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));
	ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_DefaultOpen | (select == entity ? ImGuiTreeNodeFlags_Selected : 0) | (has_child ? 0 : ImGuiTreeNodeFlags_Leaf);

	bool open = ImGui::TreeNodeEx(std::to_string(entity).c_str(), flags, "%s", tag.c_str());

	ImGui::PopStyleVar();

	// Delete entity
	ImGui::PushID(std::to_string(entity).c_str());
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
		select = entity;
	}

	// Drag and drop
	if (ImGui::BeginDragDropSource())
	{
		if (entity.HasComponent<cmpt::Hierarchy>())
		{
			ImGui::SetDragDropPayload("Entity", &entity, sizeof(Entity));
		}
		ImGui::EndDragDropSource();
	}

	if (ImGui::BeginDragDropTarget())
	{
		if (const auto *pay_load = ImGui::AcceptDragDropPayload("Entity"))
		{
			ASSERT(pay_load->DataSize == sizeof(Entity));
			SetAsSon(scene, entity, *static_cast<Entity *>(pay_load->Data));
		}
		ImGui::EndDragDropTarget();
	}

	// Recursively open children entities
	if (open)
	{
		if (has_child)
		{
			auto child = Entity(scene, entity.GetComponent<cmpt::Hierarchy>().first);
			while (child)
			{
				DrawNode(scene, child, select);
				if (child)
				{
					child = Entity(scene, child.GetComponent<cmpt::Hierarchy>().next);
				}
			}
		}
		ImGui::TreePop();
	}

	// Delete callback
	if (entity_deleted)
	{
		if (Entity(scene, entity.GetComponent<cmpt::Hierarchy>().prev))
		{
			Entity(scene, entity.GetComponent<cmpt::Hierarchy>().prev).GetComponent<cmpt::Hierarchy>().next = entity.GetComponent<cmpt::Hierarchy>().next;
		}

		if (Entity(scene, entity.GetComponent<cmpt::Hierarchy>().next))
		{
			Entity(scene, entity.GetComponent<cmpt::Hierarchy>().next).GetComponent<cmpt::Hierarchy>().prev = entity.GetComponent<cmpt::Hierarchy>().prev;
		}

		if (Entity(scene, entity.GetComponent<cmpt::Hierarchy>().parent) && entity == Entity(scene, entity.GetComponent<cmpt::Hierarchy>().parent).GetComponent<cmpt::Hierarchy>().first)
		{
			Entity(scene, entity.GetComponent<cmpt::Hierarchy>().parent).GetComponent<cmpt::Hierarchy>().first = entity.GetComponent<cmpt::Hierarchy>().next;
		}

		DeleteNode(scene, entity);

		if (select == entity)
		{
			select = entity;
		}
	}
}

template <typename T, typename Callback>
inline bool DrawComponent(const std::string &name, Scene &scene, entt::entity entity, ImGuiContext &context, Callback callback, bool static_mode = false)
{
	const ImGuiTreeNodeFlags tree_node_flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;

	Entity e(scene, entity);

	if (e.HasComponent<T>())
	{
		auto  &component                = e.GetComponent<T>();
		ImVec2 content_region_available = ImGui::GetContentRegionAvail();

		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});
		float line_height = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
		bool  open        = ImGui::TreeNodeEx((void *) typeid(T).hash_code(), tree_node_flags, name.c_str());
		ImGui::PopStyleVar();

		bool remove_component = false;
		if (!static_mode)
		{
			ImGui::SameLine(content_region_available.x - line_height * 0.5f);
			if (ImGui::Button("-", ImVec2{line_height, line_height}))
			{
				remove_component = true;
			}
		}

		bool update = false;

		if (open)
		{
			update = callback(component, context);
			ImGui::TreePop();
		}

		if (remove_component)
		{
			e.RemoveComponent<T>();
			update = true;
		}

		return update;
	}

	return false;
}

template <typename Cmpt>
inline bool DrawComponent(Scene &scene, entt::entity entity, ImGuiContext &context, bool static_mode = false)
{
	static_assert(std::is_base_of_v<cmpt::Component, Cmpt>);
	return DrawComponent<Cmpt>(
	    GetComponentName<Cmpt>(), scene, entity, context, [](Cmpt &cmpt, ImGuiContext &context) {
		    return cmpt.OnImGui(context);
	    },
	    static_mode);
}

template <typename T>
inline void AddComponents(Scene &scene, entt::entity entity)
{
	Entity e(scene, entity);
	if (!e.HasComponent<T>() && ImGui::MenuItem(typeid(T).name()))
	{
		e.AddComponent<T>();
		ImGui::CloseCurrentPopup();
	}
}

Scene::Scene(RHIDevice *device, const std::string &name) :
    p_device(device), m_name(name)
{
}

entt::registry &Scene::GetRegistry()
{
	return m_registry;
}

void Scene::Clear()
{
	m_registry.each([&](auto entity) { m_registry.destroy(entity); });
}

entt::entity Scene::Create(const std::string &name)
{
	auto      entity = m_registry.create();
	m_registry.emplace<cmpt::Tag>(entity, name);
	m_registry.emplace<cmpt::Transform>(entity);
	m_registry.emplace<cmpt::Hierarchy>(entity);
	return entity;
}

void Scene::Tick()
{
}

void Scene::OnImGui(ImGuiContext &context)
{
	// Scene Hierarchy
	{
		ImGui::Begin("Scene Hierarchy");
		ImGui::TextUnformatted(m_name.c_str());

		if (ImGui::BeginDragDropTarget())
		{
			if (const auto *pay_load = ImGui::AcceptDragDropPayload("Entity"))
			{
				ASSERT(pay_load->DataSize == sizeof(Entity));
				SetAsSon(*this, Entity(*this), *static_cast<Entity *>(pay_load->Data));
				m_update_transform = true;
			}
		}

		Entity select_entity(*this, m_select);
		m_registry.each([this, &select_entity](auto entity_id) {
			if (Entity(*this, entity_id) && Entity(*this, entity_id).GetComponent<cmpt::Hierarchy>().parent == entt::null)
			{
				DrawNode(*this, Entity(*this, entity_id), select_entity);
			}
		});
		m_select = select_entity;

		if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && ImGui::IsWindowHovered())
		{
			m_select = entt::null;
		}

		// Right-click on blank space
		if (ImGui::BeginPopupContextWindow(0, 1, false))
		{
			if (ImGui::MenuItem("New Entity"))
			{
				Create("Untitled Entity");
			}
			ImGui::EndPopup();
		}
		ImGui::End();
	}

	// Entity Inspector
	ImGui::Begin("Entity Inspector");
	if (m_registry.valid(m_select))
	{
		Entity(*this, m_select).GetComponent<cmpt::Tag>().OnImGui(context);
		ImGui::SameLine();
		ImGui::PushItemWidth(-1);

		if (ImGui::Button("Add Component"))
		{
			ImGui::OpenPopup("AddComponent");
		}
		ImGui::PopItemWidth();

		if (ImGui::BeginPopup("AddComponent"))
		{
			// AddComponents</*cmpt::Hierarchy*/>(*this, m_select);

			ImGui::EndPopup();
		}

		m_update_transform = DrawComponent<cmpt::Transform>(*this, m_select, context, true);
		DrawComponent<cmpt::Hierarchy>(*this, m_select, context, true);
	}
	ImGui::End();
}

void Scene::Save(const std::string &filename)
{
	std::ofstream os(filename, std::ios::binary);

	cereal::JSONOutputArchive archive(os);

	entt::snapshot{m_registry}
	    .entities(archive)
	    .component<
	        cmpt::Tag,
	        cmpt::Transform,
	        cmpt::Hierarchy>(archive);
}

void Scene::Load(const std::string &filename)
{
	std::ifstream os(filename, std::ios::binary);

	cereal::JSONInputArchive archive(os);

	Clear();

	entt::snapshot_loader{m_registry}
	    .entities(archive)
	    .component<
	        cmpt::Tag,
	        cmpt::Transform,
	        cmpt::Hierarchy>(archive);
}

}        // namespace Ilum