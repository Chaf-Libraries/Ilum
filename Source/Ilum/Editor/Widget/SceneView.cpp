#include "SceneView.hpp"
#include "Editor.hpp"
#include "ImGui/ImGuiHelper.hpp"

#include <CodeGeneration/Meta/SceneMeta.hpp>
#include <Core/Input.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>
#include <Scene/Component/HierarchyComponent.hpp>
#include <Scene/Component/TagComponent.hpp>
#include <Scene/Component/TransformComponent.hpp>
#include <Scene/Scene.hpp>

#include <ImGuizmo.h>
#include <imgui.h>

#include <glm/gtc/type_ptr.hpp>

namespace Ilum
{
inline glm::vec3 SmoothStep(const glm::vec3 &v1, const glm::vec3 &v2, float t)
{
	t = glm::clamp(t, 0.f, 1.f);
	t = t * t * (3.f - 2.f * t);

	glm::vec3 v = glm::mix(v1, v2, t);

	return v;
}

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

void SceneView::DisplayPresent()
{
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

		if (const auto *pay_load = ImGui::AcceptDragDropPayload("Model"))
		{
			ASSERT(pay_load->DataSize == sizeof(std::string));
			std::string uuid = *static_cast<std::string *>(pay_load->Data);

			auto *meta   = p_editor->GetRenderer()->GetResourceManager()->GetModel(uuid);
			auto  entity = p_editor->GetRenderer()->GetScene()->CreateEntity(meta->name);
			auto &cmpt   = entity.AddComponent<StaticMeshComponent>();
			cmpt.uuid    = uuid;
			if (meta)
			{
				cmpt.materials.resize(meta->submeshes.size());
				std::fill(cmpt.materials.begin(), cmpt.materials.end(), "");
			}
			p_editor->GetRenderer()->UpdateGPUScene();
		}
	}
}

void SceneView::UpdateCamera()
{
	m_camera.aspect         = m_camera.viewport.x / m_camera.viewport.y;
	m_camera.project_matrix = glm::perspective(glm::radians(m_camera.fov), m_camera.aspect, m_camera.near_plane, m_camera.far_plane);

	m_view_info.frame_count++;
	m_view_info.projection_matrix = m_camera.project_matrix;

	if (!ImGui::IsWindowFocused() || !ImGui::IsWindowHovered())
	{
		return;
	}

	if (ImGui::IsMouseDown(ImGuiMouseButton_Right))
	{
		if (!m_hide_cursor)
		{
			m_hide_cursor     = true;
			m_cursor_position = Input::GetInstance().GetMousePosition();
		}

		auto delta_time = ImGui::GetIO().DeltaTime;
		auto delta_pos  = Input::GetInstance().GetMousePosition() - m_cursor_position;
		Input::GetInstance().SetCursorPosition(m_cursor_position);
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

		if (Input::GetInstance().IsKeyPressed(KeyCode::W))
		{
			direction += forward;
		}
		if (Input::GetInstance().IsKeyPressed(KeyCode::S))
		{
			direction -= forward;
		}
		if (Input::GetInstance().IsKeyPressed(KeyCode::D))
		{
			direction += right;
		}
		if (Input::GetInstance().IsKeyPressed(KeyCode::A))
		{
			direction -= right;
		}
		if (Input::GetInstance().IsKeyPressed(KeyCode::Q))
		{
			direction += up;
		}
		if (Input::GetInstance().IsKeyPressed(KeyCode::E))
		{
			direction -= up;
		}

		// m_camera.velocity = SmoothStep(m_camera.velocity, direction, 0.2f);

		m_camera.position += direction * delta_time * m_camera.speed;

		m_view_info.position    = m_camera.position;
		m_view_info.view_matrix = glm::lookAt(m_camera.position, m_camera.position + forward, up);
		m_view_info.frame_count = 0;
	}
	else if (m_hide_cursor)
	{
		m_hide_cursor = false;
	}

	m_view_info.view_projection_matrix = m_view_info.projection_matrix * m_view_info.view_matrix;
	p_editor->GetRenderer()->SetViewInfo(m_view_info);
}

void SceneView::MoveEntity()
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
}        // namespace Ilum