#include "Transform.hpp"
#include "Camera.hpp"
#include "Hierarchy.hpp"
#include "Light.hpp"

#include "Scene/Entity.hpp"

#include <Shaders/ShaderInterop.hpp>

#include <imgui.h>
#include <imgui_internal.h>

namespace Ilum::cmpt
{
inline bool DrawVec3Control(const std::string &label, glm::vec3 &values, float resetValue = 0.0f, float columnWidth = 100.0f)
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
	update = ImGui::DragFloat("##X", &values.x, 0.1f, 0.0f, 0.0f, "%.3f") || update;
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
	update = ImGui::DragFloat("##Y", &values.y, 0.1f, 0.0f, 0.0f, "%.3f") || update;
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
	update = ImGui::DragFloat("##Z", &values.z, 0.1f, 0.0f, 0.0f, "%.3f") || update;
	ImGui::PopItemWidth();

	ImGui::PopStyleVar();

	ImGui::Columns(1);

	ImGui::PopID();

	return update;
}

bool Transform::OnImGui(ImGuiContext &context)
{
	m_update |= DrawVec3Control("Translation", m_translation, 0.f);
	m_update |= DrawVec3Control("Rotation", m_rotation, 0.f);
	m_update |= DrawVec3Control("Scale", m_scale, 1.f);
	return m_update;
}

void Transform::Tick(Scene &scene, entt::entity entity, RHIDevice *device)
{
	if (m_update)
	{
		Entity e(scene, entity);

		m_local_transform = glm::scale(glm::translate(glm::mat4(1.f), m_translation) * glm::mat4_cast(glm::qua<float>(glm::radians(m_rotation))), m_scale);

		const auto &hierachy = e.GetComponent<cmpt::Hierarchy>();

		if (hierachy.GetParent() == entt::null)
		{
			m_world_transform = m_local_transform;
		}
		else
		{
			m_world_transform = Entity(scene, hierachy.GetParent()).GetComponent<Transform>().m_world_transform * m_local_transform;
		}

		auto child = hierachy.GetFirst();
		while (child != entt::null)
		{
			Entity(scene, child).GetComponent<cmpt::Transform>().Update();
			child = Entity(scene, child).GetComponent<cmpt::Hierarchy>().GetNext();
		}

		if (e.HasComponent<cmpt::MeshRenderer>())
		{
			e.GetComponent<cmpt::MeshRenderer>().Update();
		}

		if (e.HasComponent<cmpt::Light>())
		{
			e.GetComponent<cmpt::Light>().Update();
		}

		if (scene.GetRegistry().valid(scene.GetMainCamera()))
		{
			scene.GetRegistry().get<cmpt::Camera>(scene.GetMainCamera()).Update();
		}

		m_update = false;
	}
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

const glm::mat4 &Transform::GetLocalTransform() const
{
	return m_local_transform;
}

const glm::mat4 &Transform::GetWorldTransform() const
{
	return m_world_transform;
}

void Transform::SetTranslation(const glm::vec3 &translation)
{
	m_translation = translation;
	m_update      = true;
}

void Transform::SetRotation(const glm::vec3 &rotation)
{
	m_rotation = rotation;
	m_update   = true;
}

void Transform::SetScale(const glm::vec3 &scale)
{
	m_scale  = scale;
	m_update = true;
}

}        // namespace Ilum::cmpt