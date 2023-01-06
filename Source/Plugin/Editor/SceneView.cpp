#pragma once

#include <Core/Window.hpp>
#include <Editor/Editor.hpp>
#include <Editor/Widget.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/Resource/Animation.hpp>
#include <Resource/Resource/Prefab.hpp>
#include <Resource/ResourceManager.hpp>
#include <Scene/Components/AllComponents.hpp>
#include <Scene/Node.hpp>
#include <Scene/Scene.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <imgui_internal.h>

#include <IconsFontAwesome/IconsFontAwesome5.h>

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
	struct
	{
		float speed     = 1.f;
		float sensitity = 3.f;

		glm::vec3 velocity = glm::vec3(0.f);
		glm::vec2 viewport = glm::vec2(0.f);
	} m_camera_config;

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
		if (!p_editor->GetMainCamera())
		{
			auto perspectives = p_editor->GetRenderer()->GetScene()->GetComponents<Cmpt::PerspectiveCamera>();
			if (!perspectives.empty())
			{
				p_editor->SetMainCamera(perspectives[0]);
			}
			else
			{
				auto orthographics = p_editor->GetRenderer()->GetScene()->GetComponents<Cmpt::OrthographicCamera>();
				if (!orthographics.empty())
				{
					p_editor->SetMainCamera(orthographics[0]);
				}
			}
		}

		if (!ImGui::Begin(m_name.c_str()) || !p_editor->GetMainCamera())
		{
			ShowToolBar();
			ImGui::End();
			return;
		}

		UpdateAnimation();

		ShowToolBar();

		auto *camera = p_editor->GetMainCamera();

		ImGuizmo::SetDrawlist();

		ImVec2 offset              = ImGui::GetCursorPos();
		ImVec2 scene_view_size     = ImVec2(ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x, ImGui::GetWindowContentRegionMax().y - ImGui::GetWindowContentRegionMin().y);
		ImVec2 scene_view_position = ImVec2(ImGui::GetWindowPos().x + offset.x, ImGui::GetWindowPos().y + offset.y);

		m_camera_config.viewport.x = scene_view_size.x - (static_cast<uint32_t>(scene_view_size.x) % 2 != 0 ? 1.0f : 0.0f);
		m_camera_config.viewport.y = scene_view_size.y - (static_cast<uint32_t>(scene_view_size.y) % 2 != 0 ? 1.0f : 0.0f);

		UpdateCamera();

		ImGuizmo::SetRect(scene_view_position.x, scene_view_position.y, m_camera_config.viewport.x, m_camera_config.viewport.y);

		ImGuizmo::DrawGrid(glm::value_ptr(camera->GetViewMatrix()), glm::value_ptr(camera->GetProjectionMatrix()), glm::value_ptr(glm::mat4(1.0)), 1000.f);

		DisplayPresent();

		MoveEntity();

		ImGui::End();
	}

  private:
	template <ResourceType _Ty>
	void DropTarget(Editor *editor, const std::string &name)
	{
	}

	template <>
	void DropTarget<ResourceType::Prefab>(Editor *editor, const std::string &name)
	{
		auto *prefab = editor->GetRenderer()->GetResourceManager()->Get<ResourceType::Prefab>(name);
		if (prefab)
		{
			auto &root  = prefab->GetRoot();
			auto *scene = editor->GetRenderer()->GetScene();

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

				std::vector<std::string> materials;
				std::vector<std::string> animations;
				Cmpt::Renderable        *renderable = nullptr;
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
							Cmpt::SkinnedMeshRenderer *skinned_mesh_renderer = node->HasComponent<Cmpt::SkinnedMeshRenderer>() ?
							                                                       node->GetComponent<Cmpt::SkinnedMeshRenderer>() :
							                                                       node->AddComponent<Cmpt::SkinnedMeshRenderer>(std::make_unique<Cmpt::SkinnedMeshRenderer>(node));
							skinned_mesh_renderer->AddSubmesh(uuid);
							renderable = skinned_mesh_renderer;
						}
						break;
						case ResourceType::Material: {
							materials.push_back(uuid);
						}
						break;
						case ResourceType::Animation: {
							animations.push_back(uuid);
						}
						break;
						default:
							break;
					}
				}

				if (renderable)
				{
					for (auto &uuid : materials)
					{
						renderable->AddMaterial(uuid);
					}
					for (auto &uuid : animations)
					{
						renderable->AddAnimation(uuid);
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
			DropTarget<ResourceType::Prefab>(editor, static_cast<const char *>(pay_load->Data));
		}
	}

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
	}

	void UpdateCamera()
	{
		auto *camera    = p_editor->GetMainCamera();
		auto *transform = camera->GetNode()->GetComponent<Cmpt::Transform>();

		glm::mat4 view_matrix = camera->GetViewMatrix();

		camera->SetAspect(m_camera_config.viewport.x / m_camera_config.viewport.y);

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

			float yaw   = std::atan2f(-view_matrix[2][2], -view_matrix[0][2]);
			float pitch = std::asinf(-glm::clamp(view_matrix[1][2], -1.f, 1.f));

			if (delta_pos.x != 0.f)
			{
				yaw += m_camera_config.sensitity * delta_time * delta_pos.x * 0.1f;
				glm::vec3 rotation = transform->GetRotation();
				rotation.y         = -glm::degrees(yaw) - 90.f;
				transform->SetRotation(rotation);
			}

			if (delta_pos.y != 0.f)
			{
				pitch -= m_camera_config.sensitity * delta_time * delta_pos.y * 0.1f;
				glm::vec3 rotation = transform->GetRotation();
				rotation.x         = glm::degrees(pitch);
				transform->SetRotation(rotation);
			}

			glm::vec3 forward = {};

			forward.x = glm::cos(yaw) * glm::cos(pitch);
			forward.y = glm::sin(pitch);
			forward.z = glm::sin(yaw) * glm::cos(pitch);

			forward = glm::normalize(forward);

			glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3{0.f, 1.f, 0.f}));
			glm::vec3 up    = glm::normalize(glm::cross(right, forward));

			glm::vec3 direction = glm::vec3(0.f);

			m_camera_config.speed = glm::clamp(m_camera_config.speed + 0.5f * ImGui::GetIO().MouseWheel, 0.f, 30.f);

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

			m_camera_config.velocity = SmoothStep(m_camera_config.velocity, direction * m_camera_config.speed, 0.2f);

			transform->SetTranslation(transform->GetTranslation() + delta_time * m_camera_config.velocity);
		}
		else if (m_hide_cursor)
		{
			m_hide_cursor            = false;
			m_camera_config.velocity = glm::vec3(0.f);
		}

		p_editor->GetRenderer()->UpdateView(camera);
	}

	void MoveEntity()
	{
		Node *node   = p_editor->GetSelectedNode();
		auto *camera = p_editor->GetMainCamera();

		if (!node)
		{
			return;
		}

		auto *transform = node->GetComponent<Cmpt::Transform>();

		glm::mat4 local_transform = transform->GetLocalTransform();

		if (ImGuizmo::Manipulate(
		        glm::value_ptr(camera->GetViewMatrix()),
		        glm::value_ptr(camera->GetProjectionMatrix()),
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

	void ShowToolBar()
	{
#define SHOW_TIPS(str)               \
	if (ImGui::IsItemHovered())      \
	{                                \
		ImGui::BeginTooltip();       \
		ImGui::TextUnformatted(str); \
		ImGui::EndTooltip();         \
	}

		ImGui::Indent();
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
		ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.f, 0.f));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5.f, 5.f));

		ImGui::SameLine();

		if (ImGui::Button(ICON_FA_CAMERA, ImVec2(20.f, 20.f)))
		{
			ImGui::OpenPopup("CameraPopup");
		}

		SHOW_TIPS("Camera");

		ImGui::SameLine();

		if (ImGui::Button(ICON_FA_SAVE, ImVec2(20.f, 20.f)))
		{
		}

		SHOW_TIPS("Save");

		ImGui::SameLine();

		if (ImGui::Button(m_play_animation ? ICON_FA_PAUSE : ICON_FA_PLAY, ImVec2(20.f, 20.f)))
		{
			m_play_animation = !m_play_animation;
		}

		if (m_play_animation)
		{
			m_animation_time += ImGui::GetIO().DeltaTime * 0.5f;
			if (m_animation_time > p_editor->GetRenderer()->GetMaxAnimationTime())
			{
				m_animation_time = 0.f;
			}
		}

		ImGui::SameLine();

		ImGui::PushItemWidth(200.f);
		if (ImGui::SliderFloat("Animation Time", &m_animation_time, 0.f, p_editor->GetRenderer()->GetMaxAnimationTime()) || m_play_animation)
		{
			p_editor->GetRenderer()->SetAnimationTime(m_animation_time);
		}
		ImGui::PopItemWidth();

		SHOW_TIPS("Play Animation");

		if (ImGui::BeginPopup("CameraPopup"))
		{
			auto perspective_cameras  = p_editor->GetRenderer()->GetScene()->GetComponents<Cmpt::PerspectiveCamera>();
			auto orthographic_cameras = p_editor->GetRenderer()->GetScene()->GetComponents<Cmpt::OrthographicCamera>();

			ImGui::PushItemWidth(80.f);
			ImGui::Text("Camera Properties");
			if ((!perspective_cameras.empty() || !orthographic_cameras.empty()) && ImGui::BeginMenu("Main Camera"))
			{
				for (auto &perspective_camera : perspective_cameras)
				{
					bool selected = perspective_camera == p_editor->GetMainCamera();
					if (ImGui::MenuItem(perspective_camera->GetNode()->GetName().c_str(), nullptr, &selected))
					{
						p_editor->SetMainCamera(perspective_camera);
					}
				}
				for (auto &orthographic_camera : orthographic_cameras)
				{
					bool selected = orthographic_camera == p_editor->GetMainCamera();
					if (ImGui::MenuItem(orthographic_camera->GetNode()->GetName().c_str(), nullptr, &selected))
					{
						p_editor->SetMainCamera(orthographic_camera);
					}
				}
				ImGui::EndMenu();
			}
			ImGui::DragFloat("Speed", &m_camera_config.speed, 0.01f, 0.f, std::numeric_limits<float>::max());
			ImGui::DragFloat("Sensitity", &m_camera_config.sensitity, 0.01f, 0.f, std::numeric_limits<float>::max());
			ImGui::PopItemWidth();
			ImGui::EndPopup();
		}

		ImGui::PopStyleVar();
		ImGui::PopStyleVar();
		ImGui::PopStyleVar();
		ImGui::PopStyleColor();

		ImGui::Unindent();
	}

	void UpdateAnimation()
	{
		if (m_play_animation)
		{
			m_animation_time += ImGui::GetIO().DeltaTime;

			auto *resource_manager = p_editor->GetRenderer()->GetResourceManager();
			auto  animation_names  = resource_manager->GetResources<ResourceType::Animation>();
			for (auto &animation_name : animation_names)
			{
				auto *animation = resource_manager->Get<ResourceType::Animation>(animation_name);
				animation;
			}
		}
	}

  private:
	std::map<std::string, std::unique_ptr<RHITexture>> m_icons;

	float m_animation_time = 0.f;

	bool m_play_animation = false;
};

extern "C"
{
	EXPORT_API SceneView *Create(Editor *editor, ImGuiContext *context)
	{
		ImGui::SetCurrentContext(context);
		return new SceneView(editor);
	}
}