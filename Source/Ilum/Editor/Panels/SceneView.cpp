#include "SceneView.hpp"

#include "Device/Input.hpp"
#include "Device/Window.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Scene/Component/MeshRenderer.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "Editor/Editor.hpp"

#include "ImGui/ImGuiContext.hpp"
#include "ImGui/ImGuiTool.hpp"

#include "Loader/ImageLoader/ImageLoader.hpp"

#include <SDL.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <ImGuizmo/ImGuizmo.h>

#include <imgui.h>

namespace Ilum::panel
{
SceneView::SceneView()
{
	m_name = "SceneView";

	// First update
	auto &main_camera                  = Renderer::instance()->Main_Camera;
	main_camera.front.x                = std::cosf(main_camera.pitch) * std::cosf(main_camera.yaw);
	main_camera.front.y                = std::sinf(main_camera.pitch);
	main_camera.front.z                = std::cosf(main_camera.pitch) * std::sinf(main_camera.yaw);
	main_camera.front                  = glm::normalize(main_camera.front);
	main_camera.right                  = glm::normalize(glm::cross(main_camera.front, glm::vec3{0.f, 1.f, 0.f}));
	main_camera.up                     = glm::normalize(glm::cross(main_camera.right, main_camera.front));
	main_camera.view                   = glm::lookAt(main_camera.position, main_camera.front + main_camera.position, main_camera.up);
	main_camera.camera.aspect          = static_cast<float>(Window::instance()->getWidth()) / static_cast<float>(Window::instance()->getHeight());
	main_camera.projection             = glm::perspective(glm::radians(main_camera.camera.fov),
                                              main_camera.camera.aspect,
                                              main_camera.camera.near,
                                              main_camera.camera.far);
	main_camera.camera.view_projection = main_camera.projection * main_camera.view;

	ImageLoader::loadImageFromFile(m_icons["translate"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/translate.png"));
	ImageLoader::loadImageFromFile(m_icons["rotate"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/rotate.png"));
	ImageLoader::loadImageFromFile(m_icons["scale"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/scale.png"));
	ImageLoader::loadImageFromFile(m_icons["select"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/select.png"));
	ImageLoader::loadImageFromFile(m_icons["grid"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/grid.png"));
	ImageLoader::loadImageFromFile(m_icons["transform"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/transform.png"));
}

void SceneView::draw(float delta_time)
{
	auto render_graph = Renderer::instance()->getRenderGraph();

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 5.f));

	ImGui::Begin("SceneView", &active, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
	updateMainCamera(delta_time);

	// Tool Bar
	showToolBar();

	ImGuizmo::SetDrawlist();

	auto offset              = ImGui::GetCursorPos();
	auto scene_view_size     = ImGui::GetWindowContentRegionMax() - ImGui::GetWindowContentRegionMin();
	auto scene_view_position = ImGui::GetWindowPos() + offset;

	scene_view_size.x -= static_cast<uint32_t>(scene_view_size.x) % 2 != 0 ? 1.0f : 0.0f;
	scene_view_size.y -= static_cast<uint32_t>(scene_view_size.y) % 2 != 0 ? 1.0f : 0.0f;

	onResize(VkExtent2D{static_cast<uint32_t>(scene_view_size.x), static_cast<uint32_t>(scene_view_size.y)});

	ImGuizmo::SetRect(scene_view_position.x, scene_view_position.y, scene_view_size.x, scene_view_size.y);

	if (m_grid)
	{
		ImGuizmo::DrawGrid(glm::value_ptr(Renderer::instance()->Main_Camera.view), glm::value_ptr(Renderer::instance()->Main_Camera.projection), glm::value_ptr(glm::mat4(1.0)), 100.f);
	}

	// Display main scene
	if (render_graph->hasAttachment(render_graph->view()))
	{
		ImGui::Image(ImGuiContext::textureID(render_graph->getAttachment(render_graph->view()), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), scene_view_size);
	}

	// Drag new model
	if (ImGui::BeginDragDropTarget())
	{
		if (const auto *pay_load = ImGui::AcceptDragDropPayload("Model"))
		{
			ASSERT(pay_load->DataSize == sizeof(std::string));
			auto  entity        = Scene::instance()->createEntity("New Model");
			auto &mesh_renderer = entity.addComponent<cmpt::MeshRenderer>();
			mesh_renderer.model = *static_cast<std::string *>(pay_load->Data);
			mesh_renderer.materials.resize(Renderer::instance()->getResourceCache().loadModel(*static_cast<std::string *>(pay_load->Data)).get().getSubMeshes().size());
		}
		ImGui::EndDragDropTarget();
	}

	// Guizmo operation

	auto view = Renderer::instance()->Main_Camera.view;
	ImGuizmo::ViewManipulate(
	    glm::value_ptr(view),
	    8.f,
	    ImVec2(ImGui::GetWindowPos().x + scene_view_size.x - 128, ImGui::GetWindowPos().y + offset.y),
	    ImVec2(128, 128),
	    0x10101010);

	bool is_on_guizmo = false;

	if (Editor::instance()->getSelect())
	{
		auto &transform = Editor::instance()->getSelect().getComponent<cmpt::Transform>();

		if (m_guizmo_operation)
		{
			is_on_guizmo = ImGuizmo::Manipulate(
			    glm::value_ptr(Renderer::instance()->Main_Camera.view),
			    glm::value_ptr(Renderer::instance()->Main_Camera.projection),
			    static_cast<ImGuizmo::OPERATION>(m_guizmo_operation),
			    ImGuizmo::LOCAL, glm::value_ptr(transform.local_transform), NULL, NULL, NULL, NULL);

			if (is_on_guizmo)
			{
				ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(transform.local_transform),
				                                      glm::value_ptr(transform.translation),
				                                      glm::value_ptr(transform.rotation),
				                                      glm::value_ptr(transform.scale));
				transform.update = true;
			}
		}
	}

	// We don't want camera moving while handling object transform or window is not focused
	if (ImGui::IsWindowFocused() && !is_on_guizmo)
	{
		if (view != Renderer::instance()->Main_Camera.view)
		{
			auto &main_camera                      = Renderer::instance()->Main_Camera;
			main_camera.view  = view;

			// Decompose view matrix
			//main_camera.right    = glm::vec3(main_camera.view[0][0], main_camera.view[1][0], main_camera.view[2][0]);
			//main_camera.up       = glm::vec3(main_camera.view[0][1], main_camera.view[1][1], main_camera.view[2][1]);
			//main_camera.front    = glm::vec3(main_camera.view[0][2], main_camera.view[1][2], main_camera.view[2][2]);
			//main_camera.position = glm::inverse(main_camera.view) * glm::vec4(0.f, 0.f, -1.f, 0.f);

			main_camera.yaw   = std::atan2f(main_camera.view[2][2], main_camera.view[0][2]);
			if (main_camera.yaw < 0.f)
			{
				main_camera.yaw += glm::pi<float>();
			}

			main_camera.pitch = std::asinf(-main_camera.view[1][2]);

			LOG_INFO("Yaw: {}", glm::degrees(main_camera.yaw));

			main_camera.front.x = std::cosf(main_camera.pitch) * std::cosf(main_camera.yaw);
			main_camera.front.y = std::sinf(main_camera.pitch);
			main_camera.front.z = std::cosf(main_camera.pitch) * std::sinf(main_camera.yaw);
			main_camera.front   = glm::normalize(main_camera.front);

			main_camera.right = glm::normalize(glm::cross(main_camera.front, glm::vec3{0.f, 1.f, 0.f}));
			main_camera.up    = glm::normalize(glm::cross(main_camera.right, main_camera.front));

			main_camera.view                   = glm::lookAt(main_camera.position, main_camera.front + main_camera.position, main_camera.up);
			main_camera.camera.view_projection = main_camera.projection * main_camera.view;
		}
	}

	ImGui::End();

	ImGui::PopStyleVar();
}

void SceneView::updateMainCamera(float delta_time)
{
	if (!ImGui::IsWindowFocused())
	{
		return;
	}

	if (Input::instance()->getKey(KeyCode::Click_Right))
	{
		if (!m_cursor_hidden)
		{
			m_cursor_hidden = true;
			m_last_position = Input::instance()->getMousePosition();
		}

		ImGui::SetMouseCursor(ImGuiMouseCursor_None);

		auto [delta_x, delta_y] = Input::instance()->getMouseDelta();

		Input::instance()->setMousePosition(m_last_position.first, m_last_position.second);

		auto &main_camera = Renderer::instance()->Main_Camera;

		bool update = false;

		if (delta_x != 0)
		{
			main_camera.yaw += main_camera.sensitivity * delta_time * static_cast<float>(delta_x);
			update = true;
		}

		if (delta_y != 0)
		{
			main_camera.pitch -= main_camera.sensitivity * delta_time * static_cast<float>(delta_y);
			update = true;
		}

		if (update)
		{
			main_camera.front.x = std::cosf(main_camera.pitch) * std::cosf(main_camera.yaw);
			main_camera.front.y = std::sinf(main_camera.pitch);
			main_camera.front.z = std::cosf(main_camera.pitch) * std::sinf(main_camera.yaw);
			main_camera.front   = glm::normalize(main_camera.front);

			main_camera.right = glm::normalize(glm::cross(main_camera.front, glm::vec3{0.f, 1.f, 0.f}));
			main_camera.up    = glm::normalize(glm::cross(main_camera.right, main_camera.front));
		}

		if (Input::instance()->getKey(KeyCode::W))
		{
			main_camera.position += main_camera.front * main_camera.speed * delta_time;
			update = true;
		}
		if (Input::instance()->getKey(KeyCode::S))
		{
			main_camera.position -= main_camera.front * main_camera.speed * delta_time;
			update = true;
		}

		if (Input::instance()->getKey(KeyCode::D))
		{
			main_camera.position += main_camera.right * main_camera.speed * delta_time;
			update = true;
		}
		if (Input::instance()->getKey(KeyCode::A))
		{
			main_camera.position -= main_camera.right * main_camera.speed * delta_time;
			update = true;
		}
		if (Input::instance()->getKey(KeyCode::Q))
		{
			main_camera.position = main_camera.position + main_camera.up * main_camera.speed * delta_time;
			update               = true;
		}
		if (Input::instance()->getKey(KeyCode::E))
		{
			main_camera.position -= main_camera.up * main_camera.speed * delta_time;
			update = true;
		}

		if (update)
		{
			main_camera.view                   = glm::lookAt(main_camera.position, main_camera.front + main_camera.position, main_camera.up);
			main_camera.camera.view_projection = main_camera.projection * main_camera.view;
		}
	}
	else if (m_cursor_hidden)
	{
		m_cursor_hidden = false;
	}
}

void SceneView::showToolBar()
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

	ImVec4 select_color = ImVec4(0.5f, 0.5f, 0.5f, 1.f);

	ImGui::SameLine();
	if (ImGui::ImageButton(ImGuiContext::textureID(m_icons["select"], Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
	                       ImVec2(20.f, 20.f),
	                       ImVec2(0.f, 0.f),
	                       ImVec2(1.f, 1.f),
	                       -1,
	                       m_guizmo_operation != 0 ? ImVec4(0.f, 0.f, 0.f, 0.f) : select_color))
	{
		m_guizmo_operation = 0;
	}
	SHOW_TIPS("Select")

	ImGui::SameLine();
	if (ImGui::ImageButton(ImGuiContext::textureID(m_icons["translate"], Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
	                       ImVec2(20.f, 20.f),
	                       ImVec2(0.f, 0.f),
	                       ImVec2(1.f, 1.f),
	                       -1,
	                       m_guizmo_operation != ImGuizmo::TRANSLATE ? ImVec4(0.f, 0.f, 0.f, 0.f) : select_color))
	{
		m_guizmo_operation = ImGuizmo::TRANSLATE;
	}
	SHOW_TIPS("Translation")

	ImGui::SameLine();
	if (ImGui::ImageButton(ImGuiContext::textureID(m_icons["rotate"], Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
	                       ImVec2(20.f, 20.f),
	                       ImVec2(0.f, 0.f),
	                       ImVec2(1.f, 1.f),
	                       -1,
	                       m_guizmo_operation != ImGuizmo::ROTATE ? ImVec4(0.f, 0.f, 0.f, 0.f) : select_color))
	{
		m_guizmo_operation = ImGuizmo::ROTATE;
	}
	SHOW_TIPS("Rotation")

	ImGui::SameLine();
	if (ImGui::ImageButton(ImGuiContext::textureID(m_icons["scale"], Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
	                       ImVec2(20.f, 20.f),
	                       ImVec2(0.f, 0.f),
	                       ImVec2(1.f, 1.f),
	                       -1,
	                       m_guizmo_operation != ImGuizmo::SCALE ? ImVec4(0.f, 0.f, 0.f, 0.f) : select_color))
	{
		m_guizmo_operation = ImGuizmo::SCALE;
	}
	SHOW_TIPS("Scale")

	ImGui::SameLine();
	if (ImGui::ImageButton(ImGuiContext::textureID(m_icons["transform"], Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
	                       ImVec2(20.f, 20.f),
	                       ImVec2(0.f, 0.f),
	                       ImVec2(1.f, 1.f),
	                       -1,
	                       m_guizmo_operation != ImGuizmo::UNIVERSAL ? ImVec4(0.f, 0.f, 0.f, 0.f) : select_color))
	{
		m_guizmo_operation = ImGuizmo::UNIVERSAL;
	}
	SHOW_TIPS("Transform")

	ImGui::SameLine();
	if (ImGui::ImageButton(ImGuiContext::textureID(m_icons["grid"], Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
	                       ImVec2(20.f, 20.f),
	                       ImVec2(0.f, 0.f),
	                       ImVec2(1.f, 1.f),
	                       -1,
	                       !m_grid ? ImVec4(0.f, 0.f, 0.f, 0.f) : select_color))
	{
		m_grid = !m_grid;
	}
	SHOW_TIPS("Grid")

	ImGui::PopStyleColor();
	ImGui::PopStyleVar();
	ImGui::PopStyleVar();

	ImGui::Unindent();
}

void SceneView::onResize(VkExtent2D extent)
{
	if (extent.width != Renderer::instance()->getRenderTargetExtent().width ||
	    extent.height != Renderer::instance()->getRenderTargetExtent().height)
	{
		Renderer::instance()->resizeRenderTarget(extent);

		// Reset camera aspect
		auto &main_camera = Renderer::instance()->Main_Camera;
		if (main_camera.camera.aspect != static_cast<float>(extent.width) / static_cast<float>(extent.height))
		{
			main_camera.camera.aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);

			switch (main_camera.camera.type)
			{
				case cmpt::CameraType::Orthographic:
					main_camera.projection = glm::ortho(-glm::radians(main_camera.camera.fov),
					                                    glm::radians(main_camera.camera.fov),
					                                    -glm::radians(main_camera.camera.fov) / main_camera.camera.aspect,
					                                    glm::radians(main_camera.camera.fov) / main_camera.camera.aspect,
					                                    main_camera.camera.near,
					                                    main_camera.camera.far);
					break;
				case cmpt::CameraType::Perspective:
					main_camera.projection = glm::perspective(glm::radians(main_camera.camera.fov),
					                                          main_camera.camera.aspect,
					                                          main_camera.camera.near,
					                                          main_camera.camera.far);
					break;
				default:
					break;
			}

			main_camera.camera.view_projection = main_camera.projection * main_camera.view;
		}
	}
}
}        // namespace Ilum::panel