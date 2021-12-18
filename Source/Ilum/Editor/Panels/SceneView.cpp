#include "SceneView.hpp"

#include "Device/Input.hpp"
#include "Device/Window.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Scene/Component/Camera.hpp"
#include "Scene/Component/Light.hpp"
#include "Scene/Component/Renderable.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
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

#include <ImFileDialog.h>

#include <imgui.h>

__pragma(warning(push, 0))
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
    __pragma(warning(pop))

        namespace Ilum::panel
{
	inline void drawLine(const ImVec2& p1, const ImVec2& p2, const ImVec2& offset, float width, float height)
	{
		if ((p1.x < 0.f && p2.x < 0.f) || (p1.y< 0.f && p2.y < 0.f) || (p1.x>width && p2.x>width) || (p1.y>height && p2.y>height))
		{
			return;
		}

		if (p1.x>0.f&&p2.x>0.f &&p1.y>0.f)
		{
		
		}
	}

	template <typename T>
	inline void drawGizmo(const ImVec2 &offset, const Image &icon, bool enable)
	{
		if (!enable || !Renderer::instance()->hasMainCamera())
		{
			return;
		}

		auto group = Scene::instance()->getRegistry().group<T>(entt::get<cmpt::Transform, cmpt::Tag>);

		for (const auto &entity : group)
		{
			const auto &tag   = group.template get<cmpt::Tag>(entity);
			const auto &trans = group.template get<cmpt::Transform>(entity);

			glm::quat rotation;
			glm::vec3 position;
			glm::vec3 scale;
			glm::vec3 skew;
			glm::vec4 perspective;
			glm::decompose(trans.world_transform, scale, rotation, position, skew, perspective);

			cmpt::Camera *main_camera = Renderer::instance()->Main_Camera.hasComponent<cmpt::PerspectiveCamera>() ?
			                                static_cast<cmpt::Camera *>(&Renderer::instance()->Main_Camera.getComponent<cmpt::PerspectiveCamera>()) :
			                                static_cast<cmpt::Camera *>(&Renderer::instance()->Main_Camera.getComponent<cmpt::OrthographicCamera>());

			if (Renderer::instance()->Main_Camera == entity || !main_camera->frustum.isInside(position))
			{
				continue;
			}

			glm::vec2 screen_pos = main_camera->world2Screen(position, {static_cast<float>(Renderer::instance()->getRenderTargetExtent().width), static_cast<float>(Renderer::instance()->getRenderTargetExtent().height)}, {static_cast<float>(offset.x), static_cast<float>(offset.y)});
			ImGui::SetCursorPos({screen_pos.x - 20.f, screen_pos.y - ImGui::GetFontSize() - 20.f});
			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.7f, 0.7f, 0.0f));

			ImGui::PushID(static_cast<int>(entity));
			if (ImGui::ImageButton(ImGuiContext::textureID(icon, Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), ImVec2(20.f, 20.f), ImVec2(0.f, 0.f), ImVec2(1.f, 1.f), -1, ImVec4(0.f, 0.f, 0.f, 0.f)))
			{
				Editor::instance()->select(Entity(entity));
			}
			ImGui::PopID();

			ImGui::PopStyleColor();
		}
	}

	inline void drawBoundingBoxGizmo(const ImVec2 &offset, bool enable)
	{
		if (!enable || !Renderer::instance()->hasMainCamera() || Renderer::instance()->Main_Camera == Editor::instance()->getSelect())
		{
			return;
		}

		if (Editor::instance()->getSelect() && Editor::instance()->getSelect().hasComponent<cmpt::MeshletRenderer>())
		{
			const auto &mesh_renderer = Editor::instance()->getSelect().getComponent<cmpt::MeshletRenderer>();
			const auto &trans         = Editor::instance()->getSelect().getComponent<cmpt::Transform>();

			cmpt::Camera *main_camera = Renderer::instance()->Main_Camera.hasComponent<cmpt::PerspectiveCamera>() ?
			                                static_cast<cmpt::Camera *>(&Renderer::instance()->Main_Camera.getComponent<cmpt::PerspectiveCamera>()) :
			                                static_cast<cmpt::Camera *>(&Renderer::instance()->Main_Camera.getComponent<cmpt::OrthographicCamera>());

			if (!Renderer::instance()->getResourceCache().hasModel(mesh_renderer.model))
			{
				return;
			}

			auto bbox = Renderer::instance()->getResourceCache().loadModel(mesh_renderer.model).get().bounding_box.transform(trans.world_transform);

			if (!main_camera->frustum.isInside(bbox))
			{
				return;
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
				glm::vec2 screen_pos = main_camera->world2Screen(cube_vertex[i], {static_cast<float>(Renderer::instance()->getRenderTargetExtent().width), static_cast<float>(Renderer::instance()->getRenderTargetExtent().height)}, {static_cast<float>(offset.x), static_cast<float>(offset.y)});

				cube_screen_vertex[i] = ImVec2(screen_pos.x, screen_pos.y) + ImGui::GetWindowPos();
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

	inline void drawFrustumGizmo(const ImVec2 &offset, bool enable)
	{
		if (!enable || !Renderer::instance()->hasMainCamera() || Renderer::instance()->Main_Camera == Editor::instance()->getSelect())
		{
			return;
		}

		if (Editor::instance()->getSelect() && (Editor::instance()->getSelect().hasComponent<cmpt::PerspectiveCamera>() || Editor::instance()->getSelect().hasComponent<cmpt::OrthographicCamera>()))
		{
			cmpt::Camera *main_camera = Renderer::instance()->Main_Camera.hasComponent<cmpt::PerspectiveCamera>() ?
			                                static_cast<cmpt::Camera *>(&Renderer::instance()->Main_Camera.getComponent<cmpt::PerspectiveCamera>()) :
			                                static_cast<cmpt::Camera *>(&Renderer::instance()->Main_Camera.getComponent<cmpt::OrthographicCamera>());

			cmpt::Camera *camera = Editor::instance()->getSelect().hasComponent<cmpt::PerspectiveCamera>() ?
			                           static_cast<cmpt::Camera *>(&Editor::instance()->getSelect().getComponent<cmpt::PerspectiveCamera>()) :
			                           static_cast<cmpt::Camera *>(&Editor::instance()->getSelect().getComponent<cmpt::OrthographicCamera>());

			const auto &trans = Editor::instance()->getSelect().getComponent<cmpt::Transform>();

			glm::quat rotation;
			glm::vec3 position;
			glm::vec3 scale;
			glm::vec3 skew;
			glm::vec4 perspective;
			glm::decompose(trans.world_transform, scale, rotation, position, skew, perspective);

			std::array<glm::vec3, 8> frustum_camera_vertex = {
			    glm::vec3(-1.f, -1.f, 0.f),
			    glm::vec3(1.f, -1.f, 0.f),
			    glm::vec3(1.f, 1.f, 0.f),
			    glm::vec3(-1.f, 1.f, 0.f),
			    glm::vec3(-1.f, -1.f, 1.f),
			    glm::vec3(1.f, -1.f, 1.f),
			    glm::vec3(1.f, 1.f, 1.f),
			    glm::vec3(-1.f, 1.f, 1.f),
			};

			std::vector<glm::vec3> frustum_vertex(8);

			if (Editor::instance()->getSelect().hasComponent<cmpt::PerspectiveCamera>())
			{
				camera->view                  = glm::inverse(trans.world_transform);
				cmpt::PerspectiveCamera *perspective_camera = static_cast<cmpt::PerspectiveCamera *>(camera);
				camera->projection                          = glm::perspective(glm::radians(perspective_camera->fov), perspective_camera->aspect, perspective_camera->near_plane, perspective_camera->far_plane);
				camera->view_projection       = camera->projection * camera->view;
			}
			else if (Editor::instance()->getSelect().hasComponent<cmpt::OrthographicCamera>())
			{
				camera->view                 = glm::inverse(trans.world_transform);
				cmpt::OrthographicCamera *ortho_camera = static_cast<cmpt::OrthographicCamera *>(camera);

				camera->projection      = glm::ortho(ortho_camera->left, ortho_camera->right, ortho_camera->bottom, ortho_camera->top, ortho_camera->near_plane, ortho_camera->far_plane                                             );
				camera->view_projection       = camera->projection * camera->view;
			}

			glm::mat4 inv = glm::inverse(camera->view_projection);
			for (uint32_t i = 0; i < 4; i++)
			{
				glm::vec4 near_point = inv * glm::vec4(frustum_camera_vertex[i], 1.f);
				near_point /= near_point.w;
				frustum_vertex[i] = near_point;
			}
			for (uint32_t i = 4; i < 8; i++)
			{
				glm::vec4 far_point = inv * glm::vec4(frustum_camera_vertex[i], 1.f);
				far_point /= far_point.w;
				frustum_vertex[i] = far_point;
			}

			geometry::BoundingBox bbox;
			for (uint32_t i = 0; i < 8; i++)
			{
				bbox.merge(frustum_vertex);
			}

			if (!main_camera->frustum.isInside(bbox))
			{
				return;
			}

			std::array<ImVec2, 8> frustum_screen_vertex;

			for (uint32_t i = 0; i < 8; i++)
			{
				glm::vec2 screen_pos = main_camera->world2Screen(frustum_vertex[i], {static_cast<float>(Renderer::instance()->getRenderTargetExtent().width), static_cast<float>(Renderer::instance()->getRenderTargetExtent().height)}, {static_cast<float>(offset.x), static_cast<float>(offset.y)});

				frustum_screen_vertex[i] = ImVec2(screen_pos.x, screen_pos.y) + ImGui::GetWindowPos();
			}

			auto *draw_list = ImGui::GetWindowDrawList();
			draw_list->AddLine(frustum_screen_vertex[0], frustum_screen_vertex[1], ImColor(255.f, 0.f, 0.f), 1.f);
			draw_list->AddLine(frustum_screen_vertex[1], frustum_screen_vertex[2], ImColor(255.f, 0.f, 0.f), 1.f);
			draw_list->AddLine(frustum_screen_vertex[2], frustum_screen_vertex[3], ImColor(255.f, 0.f, 0.f), 1.f);
			draw_list->AddLine(frustum_screen_vertex[3], frustum_screen_vertex[0], ImColor(255.f, 0.f, 0.f), 1.f);
			draw_list->AddLine(frustum_screen_vertex[4], frustum_screen_vertex[5], ImColor(255.f, 0.f, 0.f), 1.f);
			draw_list->AddLine(frustum_screen_vertex[5], frustum_screen_vertex[6], ImColor(255.f, 0.f, 0.f), 1.f);
			draw_list->AddLine(frustum_screen_vertex[6], frustum_screen_vertex[7], ImColor(255.f, 0.f, 0.f), 1.f);
			draw_list->AddLine(frustum_screen_vertex[7], frustum_screen_vertex[4], ImColor(255.f, 0.f, 0.f), 1.f);
			draw_list->AddLine(frustum_screen_vertex[0], frustum_screen_vertex[4], ImColor(255.f, 0.f, 0.f), 1.f);
			draw_list->AddLine(frustum_screen_vertex[1], frustum_screen_vertex[5], ImColor(255.f, 0.f, 0.f), 1.f);
			draw_list->AddLine(frustum_screen_vertex[2], frustum_screen_vertex[6], ImColor(255.f, 0.f, 0.f), 1.f);
			draw_list->AddLine(frustum_screen_vertex[3], frustum_screen_vertex[7], ImColor(255.f, 0.f, 0.f), 1.f);
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
		ImageLoader::loadImageFromFile(m_icons["save"], PROJECT_SOURCE_DIR + std::string("Asset/Texture/Icon/save.png"));

		m_gizmo = {
		    {"grid", true},
		    {"aabb", false},
		    {"light", true},
		    {"camera", true},
		    {"frustum", true},
		};
	}

	void SceneView::draw(float delta_time)
	{
		auto render_graph = Renderer::instance()->getRenderGraph();

		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 5.f));

		ImGui::Begin("SceneView", &active, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

		// Acquire main camera
		auto &camera_entity = Renderer::instance()->Main_Camera;
		if (!camera_entity || (!camera_entity.hasComponent<cmpt::PerspectiveCamera>() && !camera_entity.hasComponent<cmpt::OrthographicCamera>()))
		{
			ImGui::End();
			ImGui::PopStyleVar();
			return;
		}

		cmpt::Camera *main_camera = Renderer::instance()->Main_Camera.hasComponent<cmpt::PerspectiveCamera>() ?
		                                static_cast<cmpt::Camera *>(&Renderer::instance()->Main_Camera.getComponent<cmpt::PerspectiveCamera>()) :
		                                static_cast<cmpt::Camera *>(&Renderer::instance()->Main_Camera.getComponent<cmpt::OrthographicCamera>());

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
			ImGuizmo::DrawGrid(glm::value_ptr(main_camera->view), glm::value_ptr(main_camera->projection), glm::value_ptr(glm::mat4(1.0)), 100.f);
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
				auto &mesh_renderer = entity.addComponent<cmpt::MeshletRenderer>();
				mesh_renderer.model = *static_cast<std::string *>(pay_load->Data);
				// Setting default material
				auto &model = Renderer::instance()->getResourceCache().loadModel(mesh_renderer.model);
				for (auto &submesh : model.get().submeshes)
				{
					mesh_renderer.materials.emplace_back(createScope<material::DisneyPBR>());
					*static_cast<material::DisneyPBR *>(mesh_renderer.materials.back().get()) = submesh.material;
				}
				Editor::instance()->select(entity);
				cmpt::Renderable::update = true;
			}
			ImGui::EndDragDropTarget();
		}

		// Guizmo operation
		auto view = main_camera->view;
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
			if (view != main_camera->view)
			{
				auto &    transform         = Renderer::instance()->Main_Camera.getComponent<cmpt::Transform>();
				glm::mat4 related_transform = transform.world_transform * glm::inverse(transform.local_transform);

				main_camera->view = view;

				transform.world_transform = glm::inverse(view);
				transform.local_transform = transform.world_transform * glm::inverse(related_transform);

				ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(transform.local_transform),
				                                      glm::value_ptr(transform.translation),
				                                      glm::value_ptr(transform.rotation),
				                                      glm::value_ptr(transform.scale));
			}
		}

		// Mouse picking
		if (ImGui::IsWindowFocused() && ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGuizmo::IsOver() && !ImGuizmo::IsUsing())
		{
			auto [mouse_x, mouse_y] = Input::instance()->getMousePosition();
			auto click_pos          = ImVec2(static_cast<float>(mouse_x) - scene_view_position.x, static_cast<float>(mouse_y) - scene_view_position.y);

			// Mouse picking via ray casting
			//{
			//	auto &main_camera = Renderer::instance()->Main_Camera_;

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
			//	const auto group    = Scene::instance()->getRegistry().group<>(entt::get<cmpt::MeshletRenderer, cmpt::Transform>);
			//	group.each([&](const entt::entity &entity, const cmpt::MeshletRenderer &mesh_renderer, const cmpt::Transform &transform) {
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
				Buffer staging_buffer(static_cast<VkDeviceSize>(static_cast<size_t>(entity_id_buffer.get().getWidth() * entity_id_buffer.get().getHeight())) * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
				cmd_buffer.transferLayout(entity_id_buffer, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
				cmd_buffer.copyImageToBuffer(ImageInfo{entity_id_buffer, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0, 0}, BufferInfo{staging_buffer, 0});
				cmd_buffer.transferLayout(entity_id_buffer, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_USAGE_SAMPLED_BIT);
				cmd_buffer.end();
				cmd_buffer.submitIdle();
				std::vector<uint32_t> image_data(static_cast<size_t>(entity_id_buffer.get().getWidth() * entity_id_buffer.get().getHeight()));
				std::memcpy(image_data.data(), staging_buffer.map(), image_data.size() * sizeof(uint32_t));

				click_pos.x = glm::clamp(click_pos.x, 0.f, static_cast<float>(entity_id_buffer.get().getWidth()));
				click_pos.y = glm::clamp(click_pos.y, 0.f, static_cast<float>(entity_id_buffer.get().getHeight()));

				auto entity = Entity(static_cast<entt::entity>(image_data[static_cast<size_t>(click_pos.y) * static_cast<size_t>(entity_id_buffer.get().getWidth()) + static_cast<size_t>(click_pos.x)]));
				Editor::instance()->select(entity);

				staging_buffer.unmap();
			}
		}

		drawGizmo<cmpt::DirectionalLight>(offset, m_icons["light"], m_gizmo["light"]);
		drawGizmo<cmpt::SpotLight>(offset, m_icons["light"], m_gizmo["light"]);
		drawGizmo<cmpt::PointLight>(offset, m_icons["light"], m_gizmo["light"]);
		drawGizmo<cmpt::PerspectiveCamera>(offset, m_icons["camera"], m_gizmo["camera"]);
		drawGizmo<cmpt::OrthographicCamera>(offset, m_icons["camera"], m_gizmo["camera"]);
		drawFrustumGizmo(offset, m_gizmo["frustum"]);
		drawBoundingBoxGizmo(offset, m_gizmo["aabb"]);

		if (Editor::instance()->getSelect())
		{
			auto &transform = Editor::instance()->getSelect().getComponent<cmpt::Transform>();

			if (m_guizmo_operation)
			{
				is_on_guizmo = ImGuizmo::Manipulate(
				    glm::value_ptr(main_camera->view),
				    glm::value_ptr(main_camera->projection),
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

			cmpt::Camera *main_camera = Renderer::instance()->Main_Camera.hasComponent<cmpt::PerspectiveCamera>() ?
			                                static_cast<cmpt::Camera *>(&Renderer::instance()->Main_Camera.getComponent<cmpt::PerspectiveCamera>()) :
			                                static_cast<cmpt::Camera *>(&Renderer::instance()->Main_Camera.getComponent<cmpt::OrthographicCamera>());

			auto &camera_transform = Renderer::instance()->Main_Camera.getComponent<cmpt::Transform>();

			float yaw   = std::atan2f(-main_camera->view[2][2], -main_camera->view[0][2]);
			float pitch = std::asinf(-main_camera->view[1][2]);

			yaw += m_camera_sensitivity * delta_time * static_cast<float>(delta_x);
			pitch -= m_camera_sensitivity * delta_time * static_cast<float>(delta_y);

			camera_transform.rotation.x = glm::degrees(pitch);
			camera_transform.rotation.y = -glm::degrees(yaw) - 90.f;

			glm::vec3 forward;
			forward.x = std::cosf(pitch) * std::cosf(yaw);
			forward.y = std::sinf(pitch);
			forward.z = std::cosf(pitch) * std::sinf(yaw);
			forward   = glm::normalize(forward);

			glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3{0.f, 1.f, 0.f}));
			glm::vec3 up    = glm::normalize(glm::cross(right, forward));

			glm::vec3 direction = glm::vec3(0.f);

			if (Input::instance()->getKey(KeyCode::W))
			{
				direction += forward;
			}
			if (Input::instance()->getKey(KeyCode::S))
			{
				direction -= forward;
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

			camera_transform.translation += direction * delta_time * m_camera_speed;
			glm::mat4 related_transform      = camera_transform.world_transform * glm::inverse(camera_transform.local_transform);
			camera_transform.local_transform = glm::scale(glm::translate(glm::mat4(1.f), camera_transform.translation) * glm::mat4_cast(glm::qua<float>(glm::radians(camera_transform.rotation))), camera_transform.scale);
			camera_transform.world_transform = related_transform * camera_transform.local_transform;
			//main_camera.update = true;
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
			ImGui::Checkbox("Camera", &m_gizmo["camera"]);
			ImGui::Checkbox("AABB", &m_gizmo["aabb"]);
			ImGui::Checkbox("Frustum", &m_gizmo["frustum"]);
			ImGui::EndPopup();
		}

		if (Renderer::instance()->getRenderGraph()->hasAttachment(m_display_attachment))
		{
			ImageReference attachment_buffer = Renderer::instance()->getRenderGraph()->getAttachment(m_display_attachment);
			if (attachment_buffer.get().getFormat() == VK_FORMAT_R8G8B8A8_UNORM || attachment_buffer.get().getFormat() == VK_FORMAT_R16G16B16A16_SFLOAT)
			{
				ImGui::SameLine();
				if (ImGui::ImageButton(ImGuiContext::textureID(m_icons["save"], Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
				                       ImVec2(20.f, 20.f),
				                       ImVec2(0.f, 0.f),
				                       ImVec2(1.f, 1.f),
				                       -1,
				                       ImVec4(0.f, 0.f, 0.f, 0.f)))
				{
					ifd::FileDialog::Instance().Save("SaveScreenShotDialog", "Save ScreenShot", "Image file (*.png;*hdr){}");
				}
			}
		}

		ImGui::PopStyleColor();
		ImGui::PopStyleVar();
		ImGui::PopStyleVar();
		ImGui::PopStyleVar();

		if (ifd::FileDialog::Instance().IsDone("SaveScreenShotDialog"))
		{
			if (ifd::FileDialog::Instance().HasResult())
			{
				std::string save_path = ifd::FileDialog::Instance().GetResult().u8string();

				{
					uint32_t       pixel_size        = 0;
					ImageReference attachment_buffer = Renderer::instance()->getRenderGraph()->getAttachment(m_display_attachment);
					if (attachment_buffer.get().getFormat() == VK_FORMAT_R8G8B8A8_UNORM)
					{
						pixel_size = 4;
					}
					else if (attachment_buffer.get().getFormat() == VK_FORMAT_R16G16B16A16_SFLOAT)
					{
						pixel_size = 8;
					}

					CommandBuffer cmd_buffer;
					cmd_buffer.begin();
					Buffer staging_buffer(static_cast<VkDeviceSize>(attachment_buffer.get().getWidth() * attachment_buffer.get().getHeight()) * pixel_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
					cmd_buffer.transferLayout(attachment_buffer, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
					cmd_buffer.copyImageToBuffer(ImageInfo{attachment_buffer, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0, 0}, BufferInfo{staging_buffer, 0});
					cmd_buffer.transferLayout(attachment_buffer, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_USAGE_SAMPLED_BIT);
					cmd_buffer.end();
					cmd_buffer.submitIdle();

					int w       = static_cast<int>(attachment_buffer.get().getWidth());
					int h       = static_cast<int>(attachment_buffer.get().getHeight());
					int channel = 4;

					if (attachment_buffer.get().getFormat() == VK_FORMAT_R8G8B8A8_UNORM)
					{
						std::vector<uint8_t> data(static_cast<size_t>(attachment_buffer.get().getWidth() * attachment_buffer.get().getHeight() * pixel_size));
						std::memcpy(data.data(), staging_buffer.map(), data.size());
						staging_buffer.unmap();
						stbi_write_png((save_path + ".png").c_str(), w, h, 4, data.data(), w * 4);
					}
					else if (attachment_buffer.get().getFormat() == VK_FORMAT_R16G16B16A16_SFLOAT)
					{
						std::vector<uint16_t> raw_data(static_cast<size_t>(attachment_buffer.get().getWidth() * attachment_buffer.get().getHeight() * pixel_size));
						std::memcpy(raw_data.data(), staging_buffer.map(), raw_data.size());
						staging_buffer.unmap();
						std::vector<float> data(raw_data.size());
						for (uint32_t i = 0; i < data.size(); i++)
						{
							uint16_t frac = (raw_data[i] & 0x3ff) | 0x400;
							int32_t  exp  = ((raw_data[i] & 0x7c00) >> 10) - 25;

							if (frac == 0 && exp == 0x1f)
								data[i] = std::numeric_limits<float>::infinity();
							else if (frac || exp)
								data[i] = frac * powf(2, static_cast<float>(exp));
							else
								data[i] = 0.f;

							data[i] = (raw_data[i] & 0x8000) ? -data[i] : data[i];
						}
						stbi_write_hdr((save_path + ".hdr").c_str(), w, h, 4, reinterpret_cast<float *>(data.data()));
					}
				}
			}
			ifd::FileDialog::Instance().Close();
		}

		ImGui::Unindent();
	}

	void SceneView::onResize(VkExtent2D extent)
	{
		if (extent.width != Renderer::instance()->getRenderTargetExtent().width ||
		    extent.height != Renderer::instance()->getRenderTargetExtent().height)
		{
			Renderer::instance()->resizeRenderTarget(extent);

			// Reset camera aspect

			auto &camera_entity = Renderer::instance()->Main_Camera;

			if (camera_entity && camera_entity.hasComponent<cmpt::PerspectiveCamera>())
			{
				auto &camera = camera_entity.getComponent<cmpt::PerspectiveCamera>();
				if (camera.aspect != static_cast<float>(extent.width) / static_cast<float>(extent.height))
				{
					camera.aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
				}
			}
		}
	}
}        // namespace Ilum::panel