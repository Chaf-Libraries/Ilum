#include "SceneInspector.hpp"
#include "Editor/Editor.hpp"
#include "Editor/ImGui/ImGuiHelper.hpp"

#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>
#include <Scene/Component/AllComponent.hpp>
#include <Scene/Scene.hpp>

#include <imgui.h>
#include <imgui_internal.h>

namespace Ilum
{
template <typename T>
inline bool TDrawComponent(Entity &entity, std::function<bool(T &)> &&callback, bool static_mode = false)
{
	const ImGuiTreeNodeFlags tree_node_flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;
	if (entity.HasComponent<T>())
	{
		auto &component  = entity.GetComponent<T>();
		component.update = false;

		ImVec2 content_region_available = ImGui::GetContentRegionAvail();

		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});
		float line_height = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
		bool  open        = ImGui::TreeNodeEx((void *) typeid(T).hash_code(), tree_node_flags, rttr::type::get<T>().get_name().data());
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
			T &cmpt     = entity.GetComponent<T>();
			update      = callback(cmpt);
			cmpt.update = update;
			ImGui::TreePop();
		}

		if (remove_component)
		{
			entity.RemoveComponent<T>();
			update = true;
		}
		return update;
	}
	return false;
}

template <typename T>
inline bool DrawComponent(Editor *editor, Entity &entity, bool static_mode = false)
{
	std::function<bool(T &)> func = [&](T &t) -> bool { return ImGui::EditVariant<T>("", editor, t); };
	return TDrawComponent(entity, std::move(func), static_mode);
}

template <>
inline bool DrawComponent<HierarchyComponent>(Editor *editor, Entity &entity, bool static_mode)
{
	std::function<bool(HierarchyComponent &)> func = [&](HierarchyComponent &cmpt) -> bool {
		ImGui::Text("First: %u", cmpt.first);
		ImGui::Text("Next: %u", cmpt.next);
		ImGui::Text("Prev: %u", cmpt.prev);
		ImGui::Text("Parent: %u", cmpt.parent);
		return false;
	};

	return TDrawComponent(entity, std::move(func), static_mode);
}

template <typename T>
bool EditComponent(Entity &entity)
{
	bool update = false;
	if (entity.HasComponent<T>())
	{
		T &cmpt     = entity.GetComponent<T>();
		update      = ImGui::EditVariant(cmpt);
		cmpt.update = update;
		ImGui::Separator();
	}
	return update;
}

template <typename T>
bool AddComponent(Entity &entity)
{
	if (!entity.HasComponent<T>() && ImGui::MenuItem(rttr::type::get<T>().get_name().data()))
	{
		entity.AddComponent<T>().update = true;
		ImGui::CloseCurrentPopup();
		return true;
	}
	return false;
}

template <typename Base, typename T>
inline bool HasComponent(Entity &entity)
{
	return std::is_base_of_v<Base, T> && entity.HasComponent<T>();
}

template <typename Base, typename T1, typename T2, typename... Tn>
inline bool HasComponent(Entity &entity)
{
	return HasComponent<Base, T1>(entity) || HasComponent<Base, T2, Tn...>(entity);
}

template <typename T1, typename T2, typename... Tn>
bool DrawComponent(Editor *editor, Entity &entity, bool static_mode)
{
	bool update = false;
	update |= DrawComponent<T1>(editor, entity, static_mode);
	update |= DrawComponent<T2, Tn...>(editor, entity, static_mode);
	return update;
}

template <typename T1, typename T2, typename... Tn>
bool EditComponent(Entity &entity)
{
	bool update = false;
	update |= EditComponent<T1>(entity);
	update |= EditComponent<T2, Tn...>(entity);
	return update;
}

template <typename T>
bool AddComponents(Entity &entity)
{
	return AddComponent<T>(entity);
}

template <typename T1, typename T2, typename... Tn>
bool AddComponents(Entity &entity)
{
	bool update = false;
	update |= AddComponent<T1>(entity);
	update |= AddComponents<T2, Tn...>(entity);
	return update;
}

template <typename Base, typename T1, typename... Tn>
inline void AddComponent(Entity &entity)
{
	if (HasComponent<Base, T1, Tn...>(entity))
	{
		return;
	}

	if (ImGui::BeginMenu(rttr::type::get<Base>().get_name().to_string().c_str()))
	{
		AddComponents<T1, Tn...>(entity);
		ImGui::EndMenu();
	}
}

SceneInspector::SceneInspector(Editor *editor) :
    Widget("Scene Inspector", editor)
{
}

SceneInspector::~SceneInspector()
{
}

void SceneInspector::Tick()
{
	ImGui::Begin(m_name.c_str());

	auto entity = p_editor->GetSelectedEntity();
	if (entity.IsValid())
	{
		DrawComponent<FIXED_COMPONENTS>(p_editor, entity, true);
		DrawComponent<MESH_COMPONENTS>(p_editor, entity, false);
		DrawComponent<LIGHT_COMPONENTS>(p_editor, entity, false);

		if (ImGui::Button("Add Component"))
		{
			ImGui::OpenPopup("AddComponent");
		}

		if (ImGui::BeginPopup("AddComponent"))
		{
			AddComponent<MESH_COMPONENTS>(entity);
			AddComponent<LIGHT_COMPONENTS>(entity);
			ImGui::EndPopup();
		}
	}

	ImGui::End();
}
}        // namespace Ilum