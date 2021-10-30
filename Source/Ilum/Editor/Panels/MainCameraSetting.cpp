#include "MainCameraSetting.hpp"

#include "Renderer/Renderer.hpp"

#include <imgui.h>

namespace Ilum::panel
{
MainCameraSetting::MainCameraSetting()
{
	m_name = "Main Camera Setting";
}

void MainCameraSetting::draw(float delta_time)
{
	ImGui::Begin("Main Camera Setting", &active);

	auto &main_camera = Renderer::instance()->Main_Camera;

	bool update = false;

	static const char *const camera_type[] = {"Perspective", "Orthographic"};
	static int               select_type   = 0;

	if (ImGui::Combo("Type", &select_type, camera_type, 2))
	{
		if (main_camera.camera.type != static_cast<cmpt::CameraType>(select_type))
		{
			main_camera.camera.type = static_cast<cmpt::CameraType>(select_type);
			update                  = true;
		}
	}

	update |= ImGui::DragFloat("Speed", &main_camera.speed, 0.01f, 0.0f, 100.f, "%.2f");
	update |= ImGui::DragFloat("Sensitivity", &main_camera.sensitivity, 0.01f, 0.0f, 100.f, "%.2f");
	update |= ImGui::DragFloat("Near Plane", &main_camera.camera.near, 0.01f, 0.0f, std::numeric_limits<float>::infinity(), "%.2f");
	update |= ImGui::DragFloat("Far Plane", &main_camera.camera.far, 0.01f, 0.0f, std::numeric_limits<float>::infinity(), "%.2f");
	update |= ImGui::DragFloat("Fov", &main_camera.camera.fov, 0.01f, 0.0f, 180.f, "%.2f");

	if (update)
	{
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

	ImGui::End();
}
}        // namespace Ilum::panel