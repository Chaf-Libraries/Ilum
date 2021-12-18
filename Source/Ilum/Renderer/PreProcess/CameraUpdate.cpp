#include "CameraUpdate.hpp"

#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Profiler.hpp"

#include "Scene/Component/Camera.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>

namespace Ilum::sym
{
void CameraUpdate::run()
{
	GraphicsContext::instance()->getProfiler().beginSample("Camera Update");

	auto &camera_entity = Renderer::instance()->Main_Camera;

	if (!camera_entity || (!camera_entity.hasComponent<cmpt::PerspectiveCamera>() && !camera_entity.hasComponent<cmpt::OrthographicCamera>()))
	{
		// Select first avaliable camera component
		auto perspective_cameras = Scene::instance()->getRegistry().group<cmpt::PerspectiveCamera>(entt::get<cmpt::Tag>);
		if (perspective_cameras.size() != 0)
		{
			camera_entity = Entity(perspective_cameras.front());
		}
		else
		{
			auto orthographic_cameras = Scene::instance()->getRegistry().group<cmpt::OrthographicCamera>(entt::get<cmpt::Tag>);
			if (orthographic_cameras.size() != 0)
			{
				camera_entity = Entity(orthographic_cameras.front());
			}
			else
			{
				GraphicsContext::instance()->getProfiler().endSample("Camera Update");
				return;
			}
		}
	}

	const auto &transform = camera_entity.getComponent<cmpt::Transform>();

	CameraData *camera_data           = reinterpret_cast<CameraData *>(Renderer::instance()->Render_Buffer.Camera_Buffer.map());
	camera_data->last_view_projection = camera_data->view_projection;

	if (camera_entity.hasComponent<cmpt::PerspectiveCamera>())
	{
		auto &camera                 = camera_entity.getComponent<cmpt::PerspectiveCamera>();
		camera.view                  = glm::inverse(transform.world_transform);
		camera.projection            = glm::perspective(glm::radians(camera.fov), camera.aspect, camera.near_plane, camera.far_plane);
		camera.view_projection       = camera.projection * camera.view;
		camera.frustum               = geometry::Frustum(camera_data->view_projection);
		camera.position              = transform.world_transform[3];
		camera_data->position        = transform.world_transform[3];
		camera_data->view_projection = camera.view_projection;
		for (size_t i = 0; i < 6; i++)
		{
			camera_data->frustum[i] = glm::vec4(camera.frustum.planes[i].normal, camera.frustum.planes[i].constant);
		}
	}
	else if (camera_entity.hasComponent<cmpt::OrthographicCamera>())
	{
		auto &camera                 = camera_entity.getComponent<cmpt::OrthographicCamera>();
		camera.view                  = glm::inverse(transform.world_transform);
		camera.projection            = glm::ortho(camera.left, camera.right, camera.bottom, camera.top, camera.near_plane, camera.far_plane);
		camera.view_projection       = camera.projection * camera.view;
		camera.frustum               = geometry::Frustum(camera_data->view_projection);
		camera.position              = transform.world_transform[3];
		camera_data->position        = transform.world_transform[3];
		camera_data->view_projection = camera.view_projection;
		for (size_t i = 0; i < 6; i++)
		{
			camera_data->frustum[i] = glm::vec4(camera.frustum.planes[i].normal, camera.frustum.planes[i].constant);
		}
	}

	Renderer::instance()->Render_Buffer.Camera_Buffer.unmap();

	GraphicsContext::instance()->getProfiler().endSample("Camera Update");
}
}        // namespace Ilum::sym
