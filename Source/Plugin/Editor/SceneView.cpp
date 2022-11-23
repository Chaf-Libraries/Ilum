#pragma once

#include <Core/Window.hpp>
#include <Editor/Editor.hpp>
//#include <Editor/ImGui/ImGuiHelper.hpp>
#include <Editor/Widget.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>
#include <Scene/Component/AllComponent.hpp>
#include <Scene/Scene.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <ImGuizmo.h>
#include <imgui.h>

#include <GLFW/glfw3.h>

using namespace Ilum;

inline glm::vec3 SmoothStep(const glm::vec3 &v1, const glm::vec3 &v2, float t)
{
	t = glm::clamp(t, 0.f, 1.f);
	t = t * t * (3.f - 2.f * t);

	glm::vec3 v = glm::mix(v1, v2, t);

	return v;
}

class SceneView : public Widget
{
  private:
	ViewInfo m_view_info = {};

	struct
	{
		float fov        = 45.f;
		float aspect     = 1.f;
		float near_plane = 0.01f;
		float far_plane  = 1000.f;

		float speed     = 1.f;
		float sensitity = 1.f;

		float yaw   = 0.f;
		float pitch = 0.f;

		glm::vec2 viewport = glm::vec2(0.f);

		glm::vec3 velocity      = glm::vec3(0.f);
		glm::vec3 position      = glm::vec3(0.f);
		glm::vec3 last_position = glm::vec3(0.f);

		glm::mat4 project_matrix = glm::mat4(1.f);
	} m_camera;

	glm::vec2 m_cursor_position = glm::vec2(0.f);

	bool m_hide_cursor = false;

  public:
	SceneView(Editor *editor) :
	    Widget("Scene View", editor)
	{
		glfwInit();
	}

	virtual ~SceneView() override
	{
		glfwTerminate();
	}

	virtual void Tick() override
	{
		ImGui::Begin(m_name.c_str());

		ImGuizmo::SetDrawlist();

		auto offset              = ImGui::GetCursorPos();
		auto scene_view_size     = ImGui::GetWindowContentRegionMax() - ImGui::GetWindowContentRegionMin();
		auto scene_view_position = ImGui::GetWindowPos() + offset;

		m_camera.viewport.x = scene_view_size.x - (static_cast<uint32_t>(scene_view_size.x) % 2 != 0 ? 1.0f : 0.0f);
		m_camera.viewport.y = scene_view_size.y - (static_cast<uint32_t>(scene_view_size.y) % 2 != 0 ? 1.0f : 0.0f);

		UpdateCamera();

		ImGuizmo::SetRect(scene_view_position.x, scene_view_position.y, m_camera.viewport.x, m_camera.viewport.y);

		ImGuizmo::DrawGrid(glm::value_ptr(m_view_info.view_matrix), glm::value_ptr(m_view_info.projection_matrix), glm::value_ptr(glm::mat4(1.0)), 1000.f);

		DisplayPresent();

		MoveEntity();

		ImGui::End();
	}

  private:
	void DisplayPresent()
	{
		auto *renderer = p_editor->GetRenderer();
		renderer->SetViewport(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y);
		auto *present_texture = renderer->GetPresentTexture();

		ImGui::Image(present_texture, ImGui::GetContentRegionAvail());

		if (ImGui::BeginDragDropTarget())
		{
			if (const auto *pay_load = ImGui::AcceptDragDropPayload(rttr::type::get<ResourceType>().get_enumeration().value_to_name(ResourceType::Scene).to_string().c_str()))
			{
				size_t uuid = *static_cast<size_t *>(pay_load->Data);

				auto *resource = p_editor->GetRenderer()->GetResourceManager()->GetResource<ResourceType::Scene>(uuid);
				auto *scene    = p_editor->GetRenderer()->GetScene();

				if (resource)
				{
					resource->Load(scene);
				}
			}

			if (const auto *pay_load = ImGui::AcceptDragDropPayload(rttr::type::get<ResourceType>().get_enumeration().value_to_name(ResourceType::Model).to_string().c_str()))
			{
				size_t uuid = *static_cast<size_t *>(pay_load->Data);

				auto *resource = p_editor->GetRenderer()->GetResourceManager()->GetResource<ResourceType::Model>(uuid);
				auto  entity   = p_editor->GetRenderer()->GetScene()->CreateEntity(resource->GetName());
				auto &cmpt     = entity.AddComponent<StaticMeshComponent>();
				cmpt.uuid      = uuid;
				if (resource)
				{
					cmpt.materials.resize(resource->GetSubmeshes().size());
				}
			}
		}
	}

	void UpdateCamera()
	{
		m_camera.aspect         = m_camera.viewport.x / m_camera.viewport.y;
		m_camera.project_matrix = glm::perspective(glm::radians(m_camera.fov), m_camera.aspect, m_camera.near_plane, m_camera.far_plane);

		m_view_info.frame_count++;
		m_view_info.projection_matrix     = m_camera.project_matrix;
		m_view_info.inv_projection_matrix = glm::inverse(m_camera.project_matrix);

		if (!ImGui::IsWindowFocused() || !ImGui::IsWindowHovered())
		{
			return;
		}

		if (ImGui::IsMouseDown(ImGuiMouseButton_Right))
		{
			if (!m_hide_cursor)
			{
				m_hide_cursor     = true;
				m_cursor_position = p_editor->GetWindow()->GetMousePosition();
			}

			auto delta_time = ImGui::GetIO().DeltaTime;
			auto delta_pos  = p_editor->GetWindow()->GetMousePosition() - m_cursor_position;
			p_editor->GetWindow()->SetCursorPosition(m_cursor_position);
			ImGui::SetMouseCursor(ImGuiMouseCursor_None);

			m_camera.yaw += m_camera.sensitity * delta_time * delta_pos.x;
			m_camera.pitch -= m_camera.sensitity * delta_time * delta_pos.y;

			glm::vec3 forward = {};

			forward.x = glm::cos(glm::radians(m_camera.yaw)) * glm::cos(glm::radians(m_camera.pitch));
			forward.y = glm::sin(glm::radians(m_camera.pitch));
			forward.z = glm::sin(glm::radians(m_camera.yaw)) * glm::cos(glm::radians(m_camera.pitch));

			forward = glm::normalize(forward);

			glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3{0.f, 1.f, 0.f}));
			glm::vec3 up    = glm::normalize(glm::cross(right, forward));

			glm::vec3 direction = glm::vec3(0.f);

			m_camera.speed = glm::clamp(m_camera.speed + 0.5f * ImGui::GetIO().MouseWheel, 0.f, 30.f);

			if (p_editor->GetWindow()->IsKeyPressed(KeyCode::W))
			{
				direction += forward;
			}
			if (p_editor->GetWindow()->IsKeyPressed(KeyCode::S))
			{
				direction -= forward;
			}
			if (p_editor->GetWindow()->IsKeyPressed(KeyCode::D))
			{
				direction += right;
			}
			if (p_editor->GetWindow()->IsKeyPressed(KeyCode::A))
			{
				direction -= right;
			}
			if (p_editor->GetWindow()->IsKeyPressed(KeyCode::Q))
			{
				direction += up;
			}
			if (p_editor->GetWindow()->IsKeyPressed(KeyCode::E))
			{
				direction -= up;
			}

			m_camera.velocity = SmoothStep(m_camera.velocity, direction * m_camera.speed, 0.2f);

			m_camera.position += delta_time * m_camera.velocity;

			m_view_info.position        = m_camera.position;
			m_view_info.view_matrix     = glm::lookAt(m_camera.position, m_camera.position + forward, up);
			m_view_info.inv_view_matrix = glm::inverse(m_view_info.view_matrix);
			m_view_info.frame_count     = 0;
		}
		else if (m_hide_cursor)
		{
			m_hide_cursor     = false;
			m_camera.velocity = glm::vec3(0.f);
		}

		m_view_info.view_projection_matrix = m_view_info.projection_matrix * m_view_info.view_matrix;
		p_editor->GetRenderer()->SetViewInfo(m_view_info);
	}

	void MoveEntity()
	{
		Entity entity(p_editor->GetSelectedEntity());
		if (!entity.IsValid())
		{
			return;
		}

		auto &transform = entity.GetComponent<TransformComponent>();

		if (ImGuizmo::Manipulate(
		        glm::value_ptr(m_view_info.view_matrix),
		        glm::value_ptr(m_view_info.projection_matrix),
		        ImGuizmo::OPERATION::UNIVERSAL,
		        ImGuizmo::WORLD, glm::value_ptr(transform.local_transform), NULL, NULL, NULL, NULL))
		{
			ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(transform.local_transform),
			                                      glm::value_ptr(transform.translation),
			                                      glm::value_ptr(transform.rotation),
			                                      glm::value_ptr(transform.scale));
			transform.update = true;
		}
	}
};

extern "C"
{
	__declspec(dllexport) SceneView *Create(Editor *editor, ImGuiContext *context)
	{
		ImGui::SetCurrentContext(context);
		return new SceneView(editor);
	}
}