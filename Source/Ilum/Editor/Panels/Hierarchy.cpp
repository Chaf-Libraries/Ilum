#include "Hierarchy.hpp"

#include "Scene/Component/Hierarchy.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"

#include "Editor/Editor.hpp"

#include <imgui.h>

namespace Ilum::panel
{
inline void delete_node(Entity entity)
{
	if (!entity)
	{
		return;
	}

	auto child   = Entity(entity.getComponent<cmpt::Hierarchy>().first);
	auto sibling = child ? Entity(child.getComponent<cmpt::Hierarchy>().next) : Entity();
	while (sibling)
	{
		auto tmp = Entity(sibling.getComponent<cmpt::Hierarchy>().next);
		delete_node(sibling);
		sibling = tmp;
	}
	delete_node(child);
	entity.destroy();
}

inline bool is_parent_of(Entity lhs, Entity rhs)
{
	auto parent = Entity(rhs.getComponent<cmpt::Hierarchy>().parent);
	while (parent)
	{
		if (parent == lhs)
		{
			return true;
		}
		parent = Entity(parent.getComponent<cmpt::Hierarchy>().parent);
	}
	return false;
}

inline void set_as_son(Entity new_parent, Entity new_son)
{
	if (new_parent && is_parent_of(new_son, new_parent))
	{
		return;
	}

	auto &h2 = new_son.getComponent<cmpt::Hierarchy>();

	if (h2.next != entt::null)
	{
		Entity(h2.next).getComponent<cmpt::Hierarchy>().prev = h2.prev;
	}

	if (h2.prev != entt::null)
	{
		Entity(h2.prev).getComponent<cmpt::Hierarchy>().next = h2.next;
	}

	if (h2.parent != entt::null && new_son == Entity(h2.parent).getComponent<cmpt::Hierarchy>().first)
	{
		Entity(h2.parent).getComponent<cmpt::Hierarchy>().first = h2.next;
	}

	h2.next   = new_parent ? new_parent.getComponent<cmpt::Hierarchy>().first : entt::null;
	h2.prev   = entt::null;
	h2.parent = new_parent ? new_parent.getHandle() : entt::null;

	if (new_parent && new_parent.getComponent<cmpt::Hierarchy>().first != entt::null)
	{
		Entity(new_parent.getComponent<cmpt::Hierarchy>().first).getComponent<cmpt::Hierarchy>().prev = new_son;
	}

	if (new_parent)
	{
		new_parent.getComponent<cmpt::Hierarchy>().first = new_son;
	}
}

inline void draw_node(Entity entity)
{
	if (!entity)
	{
		return;
	}

	auto &tag = entity.getComponent<cmpt::Tag>().name;

	bool active = entity.getComponent<cmpt::Tag>().active;

	bool has_child = entity.hasComponent<cmpt::Hierarchy>() && entity.getComponent<cmpt::Hierarchy>().first != entt::null;

	// Setting up
	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));
	ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_DefaultOpen | (Editor::instance()->getSelect() == entity ? ImGuiTreeNodeFlags_Selected : 0) | (has_child ? 0 : ImGuiTreeNodeFlags_Leaf);

	// Inactive entity set to gray
	if (!active)
	{
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 0.5f));
	}

	bool open = ImGui::TreeNodeEx(std::to_string(entity).c_str(), flags, "%s", tag.c_str());

	if (!active)
	{
		ImGui::PopStyleColor();
	}

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
		Editor::instance()->select(entity);
	}

	// Drag and drop
	if (ImGui::BeginDragDropSource())
	{
		if (entity.hasComponent<cmpt::Hierarchy>())
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
			set_as_son(entity, *static_cast<Entity *>(pay_load->Data));
			entity.getComponent<cmpt::Transform>().update                                 = true;
			static_cast<Entity *>(pay_load->Data)->getComponent<cmpt::Transform>().update = true;
		}
		ImGui::EndDragDropTarget();
	}

	// Recursively open children entities
	if (open)
	{
		if (has_child)
		{
			auto child = Entity(entity.getComponent<cmpt::Hierarchy>().first);
			while (child)
			{
				draw_node(child);
				if (child)
				{
					child = Entity(child.getComponent<cmpt::Hierarchy>().next);
				}
			}
		}
		ImGui::TreePop();
	}

	// Delete callback
	if (entity_deleted)
	{
		if (Entity(entity.getComponent<cmpt::Hierarchy>().prev))
		{
			Entity(entity.getComponent<cmpt::Hierarchy>().prev).getComponent<cmpt::Hierarchy>().next = entity.getComponent<cmpt::Hierarchy>().next;
		}

		if (Entity(entity.getComponent<cmpt::Hierarchy>().next))
		{
			Entity(entity.getComponent<cmpt::Hierarchy>().next).getComponent<cmpt::Hierarchy>().prev = entity.getComponent<cmpt::Hierarchy>().prev;
		}

		if (Entity(entity.getComponent<cmpt::Hierarchy>().parent) && entity == Entity(entity.getComponent<cmpt::Hierarchy>().parent).getComponent<cmpt::Hierarchy>().first)
		{
			Entity(entity.getComponent<cmpt::Hierarchy>().parent).getComponent<cmpt::Hierarchy>().first = entity.getComponent<cmpt::Hierarchy>().next;
		}

		delete_node(entity);

		if (Editor::instance()->getSelect() == entity)
		{
			Editor::instance()->select(Entity());
		}
	}
}

Hierarchy::Hierarchy()
{
	m_name = "Hierarchy";
}

void Hierarchy::draw(float delta_time)
{
	ImGui::Begin("Scene Hierarchy", &active);

	if (ImGui::BeginDragDropTarget())
	{
		if (const auto *pay_load = ImGui::AcceptDragDropPayload("Entity"))
		{
			ASSERT(pay_load->DataSize == sizeof(Entity));
			set_as_son(Entity(), *static_cast<Entity *>(pay_load->Data));
			static_cast<Entity *>(pay_load->Data)->getComponent<cmpt::Transform>().update = true;
		}
	}

	Scene::instance()->getRegistry().each([&](auto entity_id) {
		if (Entity(entity_id) && Entity(entity_id).getComponent<cmpt::Hierarchy>().parent == entt::null)
		{
			draw_node(Entity(entity_id));
		}
	});

	if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && ImGui::IsWindowHovered())
	{
		Editor::instance()->select(Entity());
	}

	// Right-click on blank space
	if (ImGui::BeginPopupContextWindow(0, 1, false))
	{
		if (ImGui::MenuItem("New Entity"))
		{
			Scene::instance()->createEntity("Untitled Entity");
		}
		ImGui::EndPopup();
	}

	ImGui::End();
}
}        // namespace Ilum::panel