#pragma once

#include <Components/AllComponents.hpp>
#include <Core/Window.hpp>
#include <Editor/Editor.hpp>
#include <Editor/Widget.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/Resource/Prefab.hpp>
#include <Resource/ResourceManager.hpp>
#include <SceneGraph/Node.hpp>
#include <SceneGraph/Scene.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <imgui_internal.h>

#include <ImGuizmo/ImGuizmo.h>

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

	template <ResourceType _Ty>
	void DropTarget(Editor *editor, size_t uuid)
	{
	}

	template <>
	void DropTarget<ResourceType::Prefab>(Editor *editor, size_t uuid)
	{
		auto *prefab = editor->GetRenderer()->GetResourceManager()->Get<ResourceType::Prefab>(uuid);
		if (prefab)
		{
			auto                                   &root        = prefab->GetRoot();
			auto                                   *scene       = editor->GetRenderer()->GetScene();
			std::function<Node *(decltype(root) &)> create_node = [&](decltype(root) &prefab_node) {
				Node *node = scene->CreateNode(prefab_node.name);

				Cmpt::Transform *transform   = node->AddComponent<Cmpt::Transform>(std::make_unique<Cmpt::Transform>(node));
				glm::vec3        translation = glm::vec3(0.f);
				glm::vec3        rotation    = glm::vec3(0.f);
				glm::vec3        scale       = glm::vec3(0.f);
				ImGuizmo::DecomposeMatrixToComponents(
				    glm::value_ptr(prefab_node.transform),
				    glm::value_ptr(translation),
				    glm::value_ptr(rotation),
				    glm::value_ptr(scale));
				transform->SetTranslation(translation);
				transform->SetRotation(rotation);
				transform->SetScale(scale);

				std::vector<size_t> materials;
				Cmpt::Renderable*    renderable = nullptr;
				for (auto &[type, uuid] : prefab_node.resources)
				{
					switch (type)
					{
						case ResourceType::Mesh: {
							Cmpt::MeshRenderer *mesh_renderer = node->HasComponent<Cmpt::MeshRenderer>() ?
							                                        node->GetComponent<Cmpt::MeshRenderer>() :
                                                                    node->AddComponent<Cmpt::MeshRenderer>(std::make_unique<Cmpt::MeshRenderer>(node));
							mesh_renderer->AddSubmesh(uuid);
							renderable = mesh_renderer;
						}
						break;
						case ResourceType::SkinnedMesh: {
							Cmpt::SkinnedMeshRenderer *skinned_mesh_renderer = node->AddComponent<Cmpt::SkinnedMeshRenderer>(std::make_unique<Cmpt::SkinnedMeshRenderer>(node));
							skinned_mesh_renderer->AddSubmesh(uuid);
							renderable = skinned_mesh_renderer;
						}
						break;
						case ResourceType::Material: {
							materials.push_back(uuid);
						}
						break;
						default:
							break;
					}
				}

				if (renderable)
				{
					for (auto& uuid : materials)
					{
						renderable->AddMaterial(uuid);
					}
				}

				for (auto &prefab_child : prefab_node.children)
				{
					Node *child = create_node(prefab_child);
					child->SetParent(node);
				}

				return node;
			};

			create_node(root);
		}
	}

	void DropTarget(Editor *editor)
	{
		if (const auto *pay_load = ImGui::AcceptDragDropPayload("Prefab"))
		{
			DropTarget<ResourceType::Prefab>(editor, *static_cast<size_t *>(pay_load->Data));
		}
	}

	virtual void Tick() override
	{
		ImGui::Begin(m_name.c_str());

		ImGuizmo::SetDrawlist();

		ImVec2 offset              = ImGui::GetCursorPos();
		ImVec2 scene_view_size     = ImVec2(ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x, ImGui::GetWindowContentRegionMax().y - ImGui::GetWindowContentRegionMin().y);
		ImVec2 scene_view_position = ImVec2(ImGui::GetWindowPos().x + offset.x, ImGui::GetWindowPos().y + offset.y);

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
			DropTarget(p_editor);
		}

		/*if (ImGui::BeginDragDropTarget())
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
		}*/
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
		Node *node = p_editor->GetSelectedNode();
		if (!node)
		{
			return;
		}

		auto *transform = node->GetComponent<Cmpt::Transform>();

		glm::mat4 local_transform = transform->GetLocalTransform();

		if (ImGuizmo::Manipulate(
		        glm::value_ptr(m_view_info.view_matrix),
		        glm::value_ptr(m_view_info.projection_matrix),
		        ImGuizmo::OPERATION::UNIVERSAL,
		        ImGuizmo::WORLD, glm::value_ptr(local_transform), NULL, NULL, NULL, NULL))
		{
			glm::vec3 translation = glm::vec3(0.f);
			glm::vec3 rotation    = glm::vec3(0.f);
			glm::vec3 scale       = glm::vec3(0.f);

			ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(local_transform),
			                                      glm::value_ptr(translation),
			                                      glm::value_ptr(rotation),
			                                      glm::value_ptr(scale));
			transform->SetTranslation(translation);
			transform->SetRotation(rotation);
			transform->SetScale(scale);
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