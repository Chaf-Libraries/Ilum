#include "SceneInspector.hpp"
#include "Editor/Editor.hpp"
#include "Editor/ImGui/ImGuiHelper.hpp"

#include <CodeGeneration/Meta/GeometryMeta.hpp>
#include <CodeGeneration/Meta/ResourceMeta.hpp>
#include <CodeGeneration/Meta/SceneMeta.hpp>
#include <Renderer/Renderer.hpp>
#include <Scene/Component/StaticMeshComponent.hpp>
#include <Scene/Component/TagComponent.hpp>
#include <Scene/Component/TransformComponent.hpp>
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
		auto  &component                = entity.GetComponent<T>();
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
inline bool DrawComponent(Entity &entity, bool static_mode = false)
{
	std::function<bool(T &)> func = [&](T &t) -> bool { return ImGui::EditVariant<T>(t); };
	return TDrawComponent(entity, std::move(func), static_mode);
}

template <>
inline bool DrawComponent<StaticMeshComponent>(Entity &entity, bool static_mode)
{
	std::function<bool(StaticMeshComponent &)> func = [&](StaticMeshComponent &t) -> bool {
		ImGui::Text("UUID");
		ImGui::SameLine();
		if (ImGui::Button(t.uuid.c_str(), ImVec2(ImGui::GetContentRegionAvailWidth() * 0.8f, 25.f)))
		{
			t.uuid = "";
		}
		for (auto &submesh : t.submeshes)
		{
			if (ImGui::TreeNode(submesh.name.c_str()))
			{
				ImGui::Text("Material editor here");
				ImGui::TreePop();
			}
		}
		if (ImGui::BeginDragDropTarget())
		{
			if (const auto *pay_load = ImGui::AcceptDragDropPayload("Model"))
			{
				ASSERT(pay_load->DataSize == sizeof(std::string));
				t.uuid = *static_cast<std::string *>(pay_load->Data);
				ImGui::EndDragDropTarget();
				return true;
			}
			ImGui::EndDragDropTarget();
		}
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

template <typename T1, typename T2, typename... Tn>
bool DrawComponent(Entity &entity, bool static_mode)
{
	bool update = false;
	update |= DrawComponent<T1>(entity, static_mode);
	update |= DrawComponent<T2, Tn...>(entity, static_mode);
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

template <typename T1, typename T2, typename... Tn>
bool AddComponent(Entity &entity)
{
	bool update = false;
	update |= AddComponent<T1>(entity);
	update |= AddComponent<T2, Tn...>(entity);
	return update;
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
		DrawComponent<TagComponent, TransformComponent>(entity, true);
		DrawComponent<StaticMeshComponent>(entity);

		if (ImGui::Button("Add Component"))
		{
			ImGui::OpenPopup("AddComponent");
		}

		if (ImGui::BeginPopup("AddComponent"))
		{
			AddComponent<TagComponent, TransformComponent, StaticMeshComponent>(entity);
			ImGui::EndPopup();
		}
	}

	ImGui::End();
}
}        // namespace Ilum