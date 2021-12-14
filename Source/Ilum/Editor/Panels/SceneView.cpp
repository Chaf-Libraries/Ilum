#include "SceneView.hpp"

#include "Device/Input.hpp"
#include "Device/Window.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Scene/Component/MeshRenderer.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "Editor/Editor.hpp"

#include "Geometry/BoundingBox.hpp"
#include "Geometry/Ray.hpp"

#include "ImGui/ImGuiContext.hpp"
#include "ImGui/ImGuiTool.hpp"

#include "Loader/ImageLoader/ImageLoader.hpp"

#include "File/FileSystem.hpp"

#include <SDL.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <ImGuizmo/ImGuizmo.h>

#include "Scene/Component/Light.hpp"

#include <imgui.h>

namespace Ilum::panel
{
inline glm::vec2 world2screen(glm::vec3 position, glm::vec2 offset)
{
	glm::vec4 pos = Renderer::instance()->Main_Camera.view_projection * glm::vec4(position, 1.f);
	pos *= 0.5f / pos.w;
	pos += glm::vec4(0.5f, 0.5f, 0.f, 0.f);
	pos.y = 1.f - pos.y;
	pos.x *= static_cast<float>(Renderer::instance()->getRenderTargetExtent().width);
	pos.y *= static_cast<float>(Renderer::instance()->getRenderTargetExtent().height);
	pos.x += static_cast<float>(offset.x);
	pos.y += static_cast<float>(offset.y);

	return glm::vec2(pos.x, pos.y);
}

template <typename T>
inline void drawComponentGizmo(const ImVec2 &offset, const Image &icon, bool enable)
{
}

template <>
inline void drawComponentGizmo<cmpt::Light>(const ImVec2 &offset, const Image &icon, bool enable)
{
	if (!enable)
	{
		return;
	}

	auto group = Scene::instance()->getRegistry().group<cmpt::Light>(entt::get<cmpt::Transform>);

	for (const auto &entity : group)
	{
		const auto &[light, trans] = group.template get<cmpt::Light, cmpt::Transform>(entity);

		glm::quat rotation;
		glm::vec3 position;
		glm::vec3 scale;
		glm::vec3 skew;
		glm::vec4 perspective;
		glm::decompose(trans.world_transform, scale, rotation, position, skew, perspective);

		if (!Renderer::instance()->Main_Camera.frustum.isInside(position))
		{
			continue;
		}

		glm::vec2 screen_pos = world2screen(position, {static_cast<float>(offset.x), static_cast<float>(offset.y)});
		ImGui::SetCursorPos({screen_pos.x - ImGui::GetFontSize() / 2.0f, screen_pos.y - ImGui::GetFontSize() / 2.0f});
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.7f, 0.7f, 0.0f));

		ImGui::PushID(static_cast<int>(entity));
		if (ImGui::ImageButton(ImGuiContext::textureID(icon, Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
		                       ImVec2(20.f, 20.f),
		                       ImVec2(0.f, 0.f),
		                       ImVec2(1.f, 1.f),
		                       -1,
		                       ImVec4(0.f, 0.f, 0.f, 0.f)))
		{
			Editor::instance()->select(Entity(entity));
		}
		ImGui::PopID();

		ImGui::PopStyleColor();
	}
}

template <>
inline void drawComponentGizmo<geometry::BoundingBox>(const ImVec2 &offset, const Image &icon, bool enable)
{
	if (!enable)
	{
		return;
	}

	auto group = Scene::instance()->getRegistry().group<>(entt::get<cmpt::MeshRenderer, cmpt::Transform>);

	for (const auto &entity : group)
	{
		const auto &[mesh_renderer, trans] = group.template get<cmpt::MeshRenderer, cmpt::Transform>(entity);

		if (!Renderer::instance()->getResourceCache().hasModel(mesh_renderer.model))
		{
			continue;
		}

		auto bbox = Renderer::instance()->getResourceCache().loadModel(mesh_renderer.model).get().bounding_box.transform(trans.world_transform);

		if (!Renderer::instance()->Main_Camera.frustum.isInside(bbox))
		{
			continue;
		}

		std::array<glm::vec3, 8> cube_vertex = {
		    bbox.min_,
		    glm::vec3(bbox.min_.x, bbox.min_.y, bbox.max_.z),
		    glm::vec3(bbox.min_.x, bbox.max_.y, bbox.min_.z),
		    glm::vec3(bbox.min_.x, bbox.max_.y, bbox.max_.z),
		    glm::vec3(bbox.max_.x, bbox.min_.y, bbox.min_.z),
		    glm::vec3(bbox.max_.x, bbox.min_.y, bbox.max_.z),
		    glm::vec3(bbox.max_.x, bbox.max_.y, bbox.min_.z),
		    bbox.max_,
		};

		std::array<ImVec2, 8> cube_screen_vertex;

		for (uint32_t i = 0; i < 8; i++)
		{
			auto pos = world2screen(cube_vertex[i], {static_cast<float>(offset.x), static_cast<float>(offset.y)});

			cube_screen_vertex[i] = ImVec2(pos.x, pos.y) + ImGui::GetWindowPos();
		}

		auto *draw_list = ImGui::GetWindowDrawList();
		draw_list->AddLine(cube_screen_vertex[0], cube_screen_vertex[1], ImColor(255.f, 0.f, 0.f), 1.f);
		draw_list->AddLine(cube_screen_vertex[0], cube_screen_vertex[2], ImColor(255.f, 0.f, 0.f), 1.f);
		draw_list->AddLine(cube_screen_vertex[1], cube_screen_vertex[3], ImColor(255.f, 0.f, 0.f), 1.f);
		draw_list->AddLine(cube_screen_vertex[2], cube_screen_vertex[3], ImColor(255.f, 0.f, 0.f), 1.f);
		draw_list->AddLine(cube_screen_vertex[4], cube_screen_vertex[5], ImColor(255.f, 0.f, 0.f), 1.f);
		draw_list->AddLine(cube_screen_vertex[4], cube_screen_vertex[6], ImColor(255.f, 0.f, 0.f), 1.f);
		draw_list->AddLine(cube_screen_vertex[5], cube_screen_vertex[7], ImColor(255.f, 0.f, 0.f), 1.f);
		draw_list->AddLine(cube_screen_vertex[6], cube_screen_vertex[7], ImColor(255.f, 0.f, 0.f), 1.f);
		draw_list->AddLine(cube_screen_vertex[0], cube_screen_vertex[4], ImColor(255.f, 0.f, 0.f), 1.f);
		draw_list->AddLine(cube_screen_vertex[1], cube_screen_vertex[5], ImColor(255.f, 0.f, 0.f), 1.f);
		draw_list->AddLine(cube_screen_vertex[2], cube_screen_vertex[6], ImColor(255.f, 0.f, 0.f), 1.f);
		draw_list->AddLine(cube_screen_vertex[3], cube_screen_vertex[7], ImColor(255.f, 0.f, 0.f), 1.f);
	}
}

SceneView::SceneView()
{
	m_name = "SceneView";

	ImageLoader::loadImageFromFile(m_icons["translate"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/translate.png"));
	ImageLoader::loadImageFromFile(m_icons["rotate"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/rotate.png"));
	ImageLoader::loadImageFromFile(m_icons["scale"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/scale.png"));
	ImageLoader::loadImageFromFile(m_icons["select"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/select.png"));
	ImageLoader::loadImageFromFile(m_icons["transform"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/transform.png"));
	ImageLoader::loadImageFromFile(m_icons["camera"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/camera.png"));
	ImageLoader::loadImageFromFile(m_icons["viewport"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/viewport.png"));
	ImageLoader::loadImageFromFile(m_icons["light"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/light.png"));
	ImageLoader::loadImageFromFile(m_icons["gizmo"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/gizmo.png"));

	m_gizmo = {
	    {"grid", true},
	    {"aabb", false},
	    {"light", true},
	};
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

	if (m_gizmo["grid"])
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
			auto  entity        = Scene::instance()->createEntity(FileSystem::getFileName(*static_cast<std::string *>(pay_load->Data), false));
			auto &mesh_renderer = entity.addComponent<cmpt::MeshRenderer>();
			mesh_renderer.model = *static_cast<std::string *>(pay_load->Data);
			// Setting default material
			auto &model = Renderer::instance()->getResourceCache().loadModel(mesh_renderer.model);
			for (auto &submesh : model.get().submeshes)
			{
				mesh_renderer.materials.emplace_back(createScope<material::DisneyPBR>());
				*static_cast<material::DisneyPBR *>(mesh_renderer.materials.back().get()) = submesh.material;
			}
			cmpt::MeshRenderer::update = true;
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

	// We don't want camera moving while handling object transform or window is not focused
	if (ImGui::IsWindowFocused() && !ImGuizmo::IsUsing())
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
	if (ImGui::IsWindowFocused() && ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGuizmo::IsOver() && !ImGuizmo::IsUsing())
	{
		auto [mouse_x, mouse_y] = Input::instance()->getMousePosition();
		auto click_pos          = ImVec2(static_cast<float>(mouse_x) - scene_view_position.x, static_cast<float>(mouse_y) - scene_view_position.y);

		// Mouse picking via ray casting
		//{
		//	auto &main_camera = Renderer::instance()->Main_Camera;

		//	float x = (click_pos.x / scene_view_size.x) * 2.f - 1.f;
		//	float y = -((click_pos.y / scene_view_size.y) * 2.f - 1.f);

		//	glm::mat4 inv = glm::inverse(main_camera.view_projection);

		//	glm::vec4 near_point = inv * glm::vec4(x, y, 0.f, 1.f);
		//	near_point /= near_point.w;
		//	glm::vec4 far_point = inv * glm::vec4(x, y, 1.f, 1.f);
		//	far_point /= far_point.w;

		//	geometry::Ray ray;
		//	ray.origin    = main_camera.position;
		//	ray.direction = glm::normalize(glm::vec3(far_point - near_point));

		//	Editor::instance()->select(Entity());
		//	float      distance = std::numeric_limits<float>::infinity();
		//	const auto group    = Scene::instance()->getRegistry().group<>(entt::get<cmpt::MeshRenderer, cmpt::Transform>);
		//	group.each([&](const entt::entity &entity, const cmpt::MeshRenderer &mesh_renderer, const cmpt::Transform &transform) {
		//		if (!Renderer::instance()->getResourceCache().hasModel(mesh_renderer.model))
		//		{
		//			return;
		//		}
		//		auto &model        = Renderer::instance()->getResourceCache().loadModel(mesh_renderer.model);
		//		float hit_distance = ray.hit(model.get().bounding_box.transform(transform.world_transform));
		//		if (distance > hit_distance)
		//		{
		//			distance = hit_distance;
		//			Editor::instance()->select(Entity(entity));
		//		}
		//	});
		//}

		// Mouse picking via g-buffer
		{
			ImageReference entity_id_buffer = Renderer::instance()->getRenderGraph()->getAttachment("debug - entity");

			CommandBuffer cmd_buffer;
			cmd_buffer.begin();
			Buffer staging_buffer(static_cast<VkDeviceSize>(entity_id_buffer.get().getWidth() * entity_id_buffer.get().getHeight()) * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
			cmd_buffer.transferLayout(entity_id_buffer, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
			cmd_buffer.copyImageToBuffer(ImageInfo{entity_id_buffer, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0, 0}, BufferInfo{staging_buffer, 0});
			cmd_buffer.transferLayout(entity_id_buffer, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_USAGE_SAMPLED_BIT);
			cmd_buffer.end();
			cmd_buffer.submitIdle();
			std::vector<uint32_t> image_data(entity_id_buffer.get().getWidth() * entity_id_buffer.get().getHeight());
			std::memcpy(image_data.data(), staging_buffer.map(), image_data.size() * sizeof(uint32_t));

			click_pos.x = glm::clamp(click_pos.x, 0.f, static_cast<float>(entity_id_buffer.get().getWidth()));
			click_pos.y = glm::clamp(click_pos.y, 0.f, static_cast<float>(entity_id_buffer.get().getHeight()));

			auto entity = Entity(static_cast<entt::entity>(image_data[click_pos.y * entity_id_buffer.get().getWidth() + click_pos.x]));
			if (entity)
			{
				Editor::instance()->select(entity);
			}

			staging_buffer.unmap();
		}
	}

	drawComponentGizmo<cmpt::Light>(offset, m_icons["light"], m_gizmo["light"]);
	drawComponentGizmo<geometry::BoundingBox>(offset, Image(), m_gizmo["aabb"]);

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
			VkFormat format = Renderer::instance()->getRenderGraph()->getAttachment(name).getFormat();
			if (name != rg->output() && format != VK_FORMAT_R32_UINT)
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

	ImGui::SameLine();
	if (ImGui::ImageButton(ImGuiContext::textureID(m_icons["gizmo"], Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
	                       ImVec2(20.f, 20.f),
	                       ImVec2(0.f, 0.f),
	                       ImVec2(1.f, 1.f),
	                       -1,
	                       ImVec4(0.f, 0.f, 0.f, 0.f)))
	{
		ImGui::OpenPopup("GizmoPopup");
	}

	if (ImGui::BeginPopup("GizmoPopup"))
	{
		ImGui::Checkbox("Grid", &m_gizmo["grid"]);
		ImGui::Checkbox("Light", &m_gizmo["light"]);
		ImGui::Checkbox("AABB", &m_gizmo["aabb"]);
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