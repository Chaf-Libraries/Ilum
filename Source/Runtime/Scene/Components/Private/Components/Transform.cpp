#include "Transform.hpp"

#include <SceneGraph/Node.hpp>

#include <imgui.h>
#include <imgui_internal.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace Ilum
{
namespace Cmpt
{
Transform::Transform(Node *node) :
    Component("Transform", node)
{
}

void Transform::OnImGui()
{
	auto draw_vec3 =
	    [&](const std::string &label, glm::vec3 &values, float resetValue = 0.0f, float columnWidth = 100.0f) -> bool {
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
		update |= ImGui::DragFloat("##X", &values.x, 0.1f, 0.0f, 0.0f, "%.3f");
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
		update |= ImGui::DragFloat("##Y", &values.y, 0.1f, 0.0f, 0.0f, "%.3f");
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
		update |= ImGui::DragFloat("##Z", &values.z, 0.1f, 0.0f, 0.0f, "%.3f");
		ImGui::PopItemWidth();

		ImGui::PopStyleVar();

		ImGui::Columns(1);

		ImGui::PopID();

		return update;
	};

	bool update = false;

	update |= draw_vec3("Translation", m_translation, 0.f);
	update |= draw_vec3("Rotation", m_rotation, 0.f);
	update |= draw_vec3("Scale", m_scale, 1.f);

	if (update)
	{
		SetDirty();
	}
}

std::type_index Transform::GetType() const
{
	return typeid(Transform);
}

const glm::vec3 &Transform::GetTranslation() const
{
	return m_translation;
}

const glm::vec3 &Transform::GetRotation() const
{
	return m_rotation;
}

const glm::vec3 &Transform::GetScale() const
{
	return m_scale;
}

const glm::mat4 Transform::GetLocalTransform() const
{
	return glm::scale(glm::translate(glm::mat4(1.f), m_translation) * glm::mat4_cast(glm::qua<float>(glm::radians(m_rotation))), m_scale);
}

 const glm::mat4 Transform::GetWorldTransform()
{
	Update();
	return m_world_transform;
}

void Transform::SetTranslation(const glm::vec3 &translation)
{
	m_translation = translation;
	SetDirty();
}

void Transform::SetRotation(const glm::vec3 &rotation)
{
	m_rotation = rotation;
	SetDirty();
}

void Transform::SetScale(const glm::vec3 &scale)
{
	m_scale = scale;
	SetDirty();
}

void Transform::SetDirty()
{
	if (!m_dirty)
	{
		m_dirty = true;
		for (auto* child : p_node->GetChildren())
		{
			child->GetComponent<Transform>()->SetDirty();
		}
	}
}

void Transform::Update()
{
	if (!m_dirty)
	{
		return;
	}

	glm::mat4 local_matrix = GetLocalTransform();

	auto parent = GetNode()->GetParent();

	if (parent)
	{
		auto *transform   = parent->GetComponent<Transform>();
		m_world_transform = transform->GetWorldTransform() * local_matrix;
	}
	else
	{
		m_world_transform = local_matrix;
	}

	m_dirty = false;
}
}        // namespace Cmpt
}        // namespace Ilum