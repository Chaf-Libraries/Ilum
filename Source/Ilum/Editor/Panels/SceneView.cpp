#include "SceneView.hpp"

#include "Device/Input.hpp"
#include "Device/Window.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Scene/Component/MeshRenderer.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "ImGui/ImGuiContext.hpp"
#include "ImGui/ImGuiTool.hpp"

#include <SDL.h>

#include <imgui.h>

namespace Ilum::panel
{
SceneView::SceneView()
{
	m_name = "SceneView";

	auto &main_camera                  = Renderer::instance()->Main_Camera;
	main_camera.view                   = glm::lookAt(main_camera.position, main_camera.front + main_camera.position, main_camera.up);
	main_camera.camera.view_projection = main_camera.projection * main_camera.view;
	main_camera.camera.aspect          = static_cast<float>(Window::instance()->getWidth()) / static_cast<float>(Window::instance()->getHeight());
	main_camera.projection             = glm::perspective(glm::radians(main_camera.camera.fov),
                                              main_camera.camera.aspect,
                                              main_camera.camera.near,
                                              main_camera.camera.far);
}

void SceneView::draw(float delta_time)
{
	auto render_graph = Renderer::instance()->getRenderGraph();
	ImGui::Begin("SceneView");

	updateMainCamera(delta_time);

	auto region = ImGui::GetWindowContentRegionMax() - ImGui::GetWindowContentRegionMin();
	onResize(VkExtent2D{static_cast<uint32_t>(region.x), static_cast<uint32_t>(region.y)});

	if (render_graph->hasAttachment(render_graph->view()))
	{
		ImGui::Image(ImGuiContext::textureID(render_graph->getAttachment(render_graph->view()), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), region);
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

	ImGui::End();
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
			main_camera.front.x = std::cos(main_camera.pitch) * std::cos(main_camera.yaw);
			main_camera.front.y = std::sin(main_camera.pitch);
			main_camera.front.z = std::cos(main_camera.pitch) * std::sin(main_camera.yaw);
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