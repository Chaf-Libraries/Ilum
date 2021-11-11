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

	ImageLoader::loadImageFromFile(m_icons["translate"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/translate.png"));
	ImageLoader::loadImageFromFile(m_icons["rotate"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/rotate.png"));
	ImageLoader::loadImageFromFile(m_icons["scale"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/scale.png"));
	ImageLoader::loadImageFromFile(m_icons["select"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/select.png"));
	ImageLoader::loadImageFromFile(m_icons["grid"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/grid.png"));
	ImageLoader::loadImageFromFile(m_icons["transform"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/transform.png"));
	ImageLoader::loadImageFromFile(m_icons["camera"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/camera.png"));
	ImageLoader::loadImageFromFile(m_icons["viewport"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/viewport.png"));
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
	if (m_display_attachment.empty())
	{
		m_display_attachment = render_graph->view();
	}
	auto &attachment = render_graph->getAttachment(m_display_attachment);

	ImGui::Image(ImGuiContext::textureID(attachment.isDepth() ? attachment.getView(ImageViewType::Depth_Only) : attachment.getView(), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), scene_view_size);

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
			auto &main_camera = Renderer::instance()->Main_Camera;
			main_camera.view  = view;

			float yaw   = std::atan2f(-main_camera.view[2][2], -main_camera.view[0][2]);
			float pitch = std::asinf(-main_camera.view[1][2]);

			main_camera.forward.x = std::cosf(pitch) * std::cosf(yaw);
			main_camera.forward.y = std::sinf(pitch);
			main_camera.forward.z = std::cosf(pitch) * std::sinf(yaw);
			main_camera.forward   = glm::normalize(main_camera.forward);

			main_camera.update = true;
		}
	}

	// Mouse picking
	if (ImGui::IsWindowFocused() && ImGui::IsWindowHovered() && Input::instance()->getKey(KeyCode::Click_Left))
	{
		auto [mouse_x, mouse_y] = Input::instance()->getMousePosition();
		auto click_pos          = ImVec2(static_cast<float>(mouse_x) - scene_view_position.x, static_cast<float>(mouse_y) - scene_view_position.y);

		auto &main_camera = Renderer::instance()->Main_Camera;

		float x = (click_pos.x / scene_view_size.x) * 2.f - 1.f;
		float y = (click_pos.y / scene_view_size.y) * 2.f - 1.f;

		glm::mat4 inv = glm::inverse(main_camera.view_projection);
		
		glm::vec4 test = main_camera.view_projection * glm::vec4(main_camera.position + 10.1f * main_camera.forward, 1.f);
		LOG_INFO("test {}, {}, {}, {}", test.x, test.y, test.z, test.w);

		glm::vec3 near_point = inv * glm::vec4(x, y, 0.f, 10.f) * main_camera.near_plane;
		glm::vec3 far_point  = inv * glm::vec4(x, y, 1.f, 1.f) * main_camera.far_plane;

		geometry::Ray ray;

		ray.origin    = near_point;
		ray.direction = glm::normalize(glm::vec3(far_point) - ray.origin);

		LOG_INFO("{}, {}", x, y);
		LOG_INFO("near({}, {}, {}), far({}, {}, {}), length: {}", near_point.x, near_point.y, near_point.z, far_point.x, far_point.y, far_point.z, glm::length(far_point-near_point));
		//auto       ray         = main_camera.genRay(click_pos.x / scene_view_size.x, click_pos.y / scene_view_size.y);
		//LOG_INFO("origin: ({}, {}, {}), direction: ({}, {}, {})", ray.origin.x, ray.origin.y, ray.origin.z, ray.direction.x, ray.direction.y, ray.direction.z);

		const auto group = Scene::instance()->getRegistry().group<>(entt::get<cmpt::MeshRenderer, cmpt::Transform>);
		group.each([&](const cmpt::MeshRenderer &mesh_renderer, const cmpt::Transform &transform) {
			auto &model = Renderer::instance()->getResourceCache().loadModel(mesh_renderer.model);
			auto  bbox  = model.get().getBoundingBox().transform(transform.world_transform);
			LOG_INFO(ray.hit(bbox));
		});
	}

	ImGui::End();

	ImGui::PopStyleVar();
}

void SceneView::updateMainCamera(float delta_time)
{
	// TODO: Better camera movement, Model view camera, Ortho camera
	if (!ImGui::IsWindowFocused() || !ImGui::IsWindowHovered())
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

		float yaw   = std::atan2f(main_camera.forward.z, main_camera.forward.x);
		float pitch = std::asinf(main_camera.forward.y);

		if (delta_x != 0)
		{
			yaw += m_camera_sensitivity * delta_time * static_cast<float>(delta_x);
			update = true;
		}

		if (delta_y != 0)
		{
			pitch -= m_camera_sensitivity * delta_time * static_cast<float>(delta_y);
			pitch  = glm::clamp(pitch, glm::radians(-89.f), glm::radians(89.f));
			update = true;
		}

		main_camera.forward.x = std::cosf(pitch) * std::cosf(yaw);
		main_camera.forward.y = std::sinf(pitch);
		main_camera.forward.z = std::cosf(pitch) * std::sinf(yaw);
		main_camera.forward   = glm::normalize(main_camera.forward);

		glm::vec3 right = glm::normalize(glm::cross(main_camera.forward, glm::vec3{0.f, 1.f, 0.f}));
		glm::vec3 up    = glm::normalize(glm::cross(right, main_camera.forward));

		glm::vec3 direction = glm::vec3(0.f);

		if (Input::instance()->getKey(KeyCode::W))
		{
			direction += main_camera.forward;
		}
		if (Input::instance()->getKey(KeyCode::S))
		{
			direction -= main_camera.forward;
		}
		if (Input::instance()->getKey(KeyCode::D))
		{
			direction += right;
		}
		if (Input::instance()->getKey(KeyCode::A))
		{
			direction -= right;
		}
		if (Input::instance()->getKey(KeyCode::Q))
		{
			direction += up;
		}
		if (Input::instance()->getKey(KeyCode::E))
		{
			direction -= up;
		}

		main_camera.position += direction * delta_time * m_camera_speed;

		main_camera.update = true;
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
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5.f, 5.f));

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

	ImGui::SameLine();
	if (ImGui::ImageButton(ImGuiContext::textureID(m_icons["camera"], Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
	                       ImVec2(20.f, 20.f),
	                       ImVec2(0.f, 0.f),
	                       ImVec2(1.f, 1.f),
	                       -1,
	                       ImVec4(0.f, 0.f, 0.f, 0.f)))
	{
		ImGui::OpenPopup("CameraSettingPopup");
	}
	if (ImGui::BeginPopup("CameraSettingPopup"))
	{
		// Camera setting
		auto &main_camera = Renderer::instance()->Main_Camera;

		static const char *const camera_type[] = {"Perspective", "Orthographic"};
		static int               select_type   = 0;

		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2.f, 2.f));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.f, 5.f));
		if (ImGui::Combo("Type", &select_type, camera_type, 2))
		{
			if (main_camera.type != static_cast<Camera::Type>(select_type))
			{
				main_camera.type   = static_cast<Camera::Type>(select_type);
				main_camera.update = true;
			}
		}

		main_camera.update |= ImGui::DragFloat("Speed", &m_camera_speed, 0.01f, 0.0f, 100.f, "%.2f");
		main_camera.update |= ImGui::DragFloat("Sensitivity", &m_camera_sensitivity, 0.01f, 0.0f, 100.f, "%.2f");
		main_camera.update |= ImGui::DragFloat("Near Plane", &main_camera.near_plane, 1.f, 0.0f, std::numeric_limits<float>::infinity(), "%.2f");
		main_camera.update |= ImGui::DragFloat("Far Plane", &main_camera.far_plane, 1.f, 0.0f, std::numeric_limits<float>::infinity(), "%.2f");
		main_camera.update |= ImGui::DragFloat("Fov", &main_camera.fov, 0.1f, 0.0f, 180.f, "%.2f");

		ImGui::PopStyleVar();
		ImGui::PopStyleVar();

		ImGui::EndPopup();
	}

	SHOW_TIPS("Camera")

	ImGui::SameLine();
	if (ImGui::ImageButton(ImGuiContext::textureID(m_icons["viewport"], Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
	                       ImVec2(20.f, 20.f),
	                       ImVec2(0.f, 0.f),
	                       ImVec2(1.f, 1.f),
	                       -1,
	                       ImVec4(0.f, 0.f, 0.f, 0.f)))
	{
		ImGui::OpenPopup("ViewportPopup");
	}

	if (ImGui::BeginPopup("ViewportPopup"))
	{
		auto *rg = Renderer::instance()->getRenderGraph();

		std::unordered_map<std::string, bool> select_display;
		for (auto &[name, image] : rg->getAttachments())
		{
			if (name != rg->output())
			{
				select_display[name] = name == m_display_attachment;
			}
		}

		for (auto &[name, select] : select_display)
		{
			if (ImGui::MenuItem(name.c_str(), NULL, &select))
			{
				if (select)
				{
					select_display[m_display_attachment] = false;
					m_display_attachment                 = name;
				}
			}
		}

		ImGui::EndPopup();
	}

	ImGui::PopStyleColor();
	ImGui::PopStyleVar();
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
		if (main_camera.aspect != static_cast<float>(extent.width) / static_cast<float>(extent.height))
		{
			main_camera.aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
			main_camera.update = true;
		}
	}
}
}        // namespace Ilum::panel