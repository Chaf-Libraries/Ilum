#include "Inspector.hpp"

#include "Editor/Editor.hpp"

#include "Scene/Component/Camera.hpp"
#include "Scene/Component/Hierarchy.hpp"
#include "Scene/Component/Light.hpp"
#include "Scene/Component/MeshRenderer.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"

#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_internal.h>

namespace Ilum::panel
{
inline bool draw_vec3_control(const std::string &label, glm::vec3 &values, float resetValue = 0.0f, float columnWidth = 100.0f)
{
	ImGuiIO &io        = ImGui::GetIO();
	auto     bold_font = io.Fonts->Fonts[0];
	bool     update    = false;

	ImGui::PushID(label.c_str());

	ImGui::Columns(2);
	ImGui::SetColumnWidth(0, columnWidth);
	ImGui::Text(label.c_str());
	ImGui::NextColumn();

	ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});

	float  line_height = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
	ImVec2 button_size = {line_height + 3.0f, line_height};

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
	ImGui::PushFont(bold_font);
	if (ImGui::Button("X", button_size))
	{
		values.x = resetValue;
		update   = true;
	}
	ImGui::PopFont();
	ImGui::PopStyleColor(3);

	ImGui::SameLine();
	update = update | ImGui::DragFloat("##X", &values.x, 0.1f, 0.0f, 0.0f, "%.2f");
	ImGui::PopItemWidth();
	ImGui::SameLine();

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.2f, 0.7f, 0.2f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.3f, 0.8f, 0.3f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.2f, 0.7f, 0.2f, 1.0f});
	ImGui::PushFont(bold_font);
	if (ImGui::Button("Y", button_size))
	{
		values.y = resetValue;
		update   = true;
	}
	ImGui::PopFont();
	ImGui::PopStyleColor(3);

	ImGui::SameLine();
	update = update | ImGui::DragFloat("##Y", &values.y, 0.1f, 0.0f, 0.0f, "%.2f");
	ImGui::PopItemWidth();
	ImGui::SameLine();

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.1f, 0.25f, 0.8f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.2f, 0.35f, 0.9f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.1f, 0.25f, 0.8f, 1.0f});
	ImGui::PushFont(bold_font);
	if (ImGui::Button("Z", button_size))
	{
		values.z = resetValue;
		update   = true;
	}
	ImGui::PopFont();
	ImGui::PopStyleColor(3);

	ImGui::SameLine();
	update = update | ImGui::DragFloat("##Z", &values.z, 0.1f, 0.0f, 0.0f, "%.2f");
	ImGui::PopItemWidth();

	ImGui::PopStyleVar();

	ImGui::Columns(1);

	ImGui::PopID();

	return update;
}

template <typename T, typename Callback>
inline void draw_component(const std::string &name, Entity entity, Callback callback, bool static_mode = false)
{
	const ImGuiTreeNodeFlags tree_node_flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;
	if (entity.hasComponent<T>())
	{
		auto & component                = entity.getComponent<T>();
		ImVec2 content_region_available = ImGui::GetContentRegionAvail();

		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});
		float line_height = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
		ImGui::Separator();
		bool open = ImGui::TreeNodeEx((void *) typeid(T).hash_code(), tree_node_flags, name.c_str());
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

		if (open)
		{
			callback(component);
			ImGui::TreePop();
		}

		if (remove_component)
		{
			entity.removeComponent<T>();
		}
	}
}

template <typename T>
inline void add_component()
{
	if (!Editor::instance()->getSelect().hasComponent<T>() && ImGui::MenuItem(typeid(T).name()))
	{
		Editor::instance()->getSelect().addComponent<T>();
		ImGui::CloseCurrentPopup();
	}
}

template <typename T1, typename T2, typename... Tn>
inline void add_component()
{
	add_component<T1>();
	add_component<T2, Tn...>();
}

template <typename T>
inline void draw_component(Entity entity)
{
}

template <typename T1, typename T2, typename... Tn>
inline void draw_component(Entity entity)
{
	draw_component<T1>(entity);
	draw_component<T2, Tn...>(entity);
}

template <>
inline void draw_component<cmpt::Tag>(Entity entity)
{
	if (entity.hasComponent<cmpt::Tag>())
	{
		ImGui::Text("Tag");

		ImGui::SameLine();
		auto &tag = entity.getComponent<cmpt::Tag>().name;
		char  buffer[64];
		memset(buffer, 0, sizeof(buffer));
		std::memcpy(buffer, tag.data(), sizeof(buffer));
		ImGui::PushItemWidth(150.f);
		if (ImGui::InputText("##Tag", buffer, sizeof(buffer)))
		{
			tag = std::string(buffer);
		}
		ImGui::PopItemWidth();
		ImGui::SameLine();
		ImGui::Checkbox("Active", &entity.getComponent<cmpt::Tag>().active);
	}
}

template <>
inline void draw_component<cmpt::Transform>(Entity entity)
{
	draw_component<cmpt::Transform>(
	    "Transform", entity, [](auto &component) {
		    component.update =
		        draw_vec3_control("Scale", component.scale, 1.f) |
		        draw_vec3_control("Translation", component.translation, 0.f) |
		        draw_vec3_control("Rotation", component.rotation, 0.f);
	    },
	    true);
}

template <>
inline void draw_component<cmpt::Hierarchy>(Entity entity)
{
	draw_component<cmpt::Hierarchy>(
	    "Hierarchy", entity, [](auto &component) {
		    ImGui::Text("Parent: %s", component.parent == entt::null ? "false" : "true");
		    ImGui::Text("Children: %s", component.first == entt::null ? "false" : "true");
		    ImGui::Text("Siblings: %s", component.next == entt::null && component.prev == entt::null ? "false" : "true");
	    },
	    true);
}

template <>
inline void draw_component<cmpt::MeshRenderer>(Entity entity)
{
	draw_component<cmpt::MeshRenderer>(
	    "MeshRenderer", entity, [](cmpt::MeshRenderer &component) {
		    ImGui::Text("Model: ");
		    ImGui::SameLine();
		    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.f, 0.f));
			if (ImGui::Button(component.model.c_str(), ImVec2(250.f, 0.f)))
			{
			    component.model = "";
			}
		    ImGui::PopStyleVar();
		    if (ImGui::BeginDragDropTarget())
		    {
			    if (const auto *pay_load = ImGui::AcceptDragDropPayload("Model"))
			    {
				    ASSERT(pay_load->DataSize == sizeof(std::string));
				    std::string new_model = *static_cast<std::string *>(pay_load->Data);
				    if (component.model != new_model)
				    {
					    component.model = new_model;
					    // TODO: Reset materials
				    }
			    }
			    ImGui::EndDragDropTarget();
		    }

		    ImGui::Text("Materials: ");
	    });
}

template <>
inline void draw_component<cmpt::Camera>(Entity entity)
{
	draw_component<cmpt::Camera>(
	    "Camera", entity, [](auto &component) {

	    });
}

template <>
inline void draw_component<cmpt::Light>(Entity entity)
{
	draw_component<cmpt::Light>(
	    "Light", entity, [](auto &component) {

	    });
}

Inspector::Inspector()
{
	m_name = "Inspector";
}

void Inspector::draw()
{
	ImGui::Begin("Inspector", &active);

	auto entity = Editor::instance()->getSelect();

	if (!entity.valid())
	{
		ImGui::End();
		return;
	}

	// Editable tag
	draw_component<cmpt::Tag>(entity);

	ImGui::SameLine();
	ImGui::PushItemWidth(-1);

	// Add components popup
	if (ImGui::Button("Add Component"))
	{
		ImGui::OpenPopup("AddComponent");
	}

	if (ImGui::BeginPopup("AddComponent"))
	{
		add_component<cmpt::MeshRenderer, cmpt::Camera, cmpt::Light>();
		ImGui::EndPopup();
	}
	ImGui::PopItemWidth();

	draw_component<cmpt::Transform, cmpt::Hierarchy, cmpt::MeshRenderer, cmpt::Camera, cmpt::Light>(entity);

	ImGui::End();
}
}        // namespace Ilum::panel