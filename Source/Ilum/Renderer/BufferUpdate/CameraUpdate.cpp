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

#define HALTION_SAMPLES 16

namespace Ilum::sym
{
CameraUpdate::CameraUpdate()
{
	m_halton_sequence = {
	    glm::vec2{0.500000, 0.333333},
	    glm::vec2{0.250000, 0.666667},
	    glm::vec2{0.750000, 0.111111},
	    glm::vec2{0.125000, 0.444444},
	    glm::vec2{0.625000, 0.777778},
	    glm::vec2{0.375000, 0.222222},
	    glm::vec2{0.875000, 0.555556},
	    glm::vec2{0.062500, 0.888889},
	    glm::vec2{0.562500, 0.037037},
	    glm::vec2{0.312500, 0.370370},
	    glm::vec2{0.812500, 0.703704},
	    glm::vec2{0.187500, 0.148148},
	    glm::vec2{0.687500, 0.481481},
	    glm::vec2{0.437500, 0.814815},
	    glm::vec2{0.937500, 0.259259},
	    glm::vec2{0.031250, 0.592593}};
}

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
			camera_entity                                                = Entity(perspective_cameras.front());
			auto extent                                                  = Renderer::instance()->getViewportExtent();
			camera_entity.getComponent<cmpt::PerspectiveCamera>().aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
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

	CameraData  *camera_data  = reinterpret_cast<CameraData *>(Renderer::instance()->Render_Buffer.Camera_Buffer.map());
	CullingData *culling_data = reinterpret_cast<CullingData *>(Renderer::instance()->Render_Buffer.Culling_Buffer.map());
	culling_data->last_view   = culling_data->view;

	if (camera_entity.hasComponent<cmpt::PerspectiveCamera>())
	{
		auto &camera = camera_entity.getComponent<cmpt::PerspectiveCamera>();

		camera.view            = glm::inverse(transform.world_transform);
		camera.projection      = glm::perspective(glm::radians(camera.fov), camera.aspect, camera.near_plane, camera.far_plane);
		camera.view_projection = camera.projection * camera.view;
		camera.frustum         = geometry::Frustum(camera_data->view_projection);
		camera.position        = transform.world_transform[3];

		culling_data->view  = camera.view;
		culling_data->P00   = camera.projection[0][0];
		culling_data->P11   = camera.projection[1][1];
		culling_data->znear = camera.near_plane;
		culling_data->zfar  = camera.far_plane;

		camera_data->position = transform.world_transform[3];
		camera_data->last_view_projection = camera.last_view_projection;
		camera_data->view_inverse         = transform.world_transform;
		camera_data->projection_inverse   = glm::inverse(camera.projection);
		camera_data->frame_num            = camera.frame_count;

		// Jitter camera
		auto jitter_offset       = m_halton_sequence[(m_frame_count++) % 16];
		camera_data->prev_jitter = camera_data->jitter;
		camera_data->jitter.x    = (2.f * jitter_offset[0] - 1.f) / static_cast<float>(Renderer::instance()->getRenderTargetExtent().width);
		camera_data->jitter.y    = (2.f * jitter_offset[1] - 1.f) / static_cast<float>(Renderer::instance()->getRenderTargetExtent().height);
		glm::mat4 projection     = camera.projection;
		projection[2][0] += camera_data->jitter.x;
		projection[2][1] += camera_data->jitter.y;
		camera_data->view_projection = projection * camera.view;

		for (size_t i = 0; i < 6; i++)
		{
			camera_data->frustum[i] = glm::vec4(camera.frustum.planes[i].normal, camera.frustum.planes[i].constant);
		}

		camera.last_view_projection = camera.view_projection;
		camera.frame_count++;
	}
	else if (camera_entity.hasComponent<cmpt::OrthographicCamera>())
	{
		auto &camera = camera_entity.getComponent<cmpt::OrthographicCamera>();

		camera.view            = glm::inverse(transform.world_transform);
		camera.projection      = glm::ortho(camera.left, camera.right, camera.bottom, camera.top, camera.near_plane, camera.far_plane);
		camera.view_projection = camera.projection * camera.view;
		camera.frustum         = geometry::Frustum(camera_data->view_projection);
		camera.position        = transform.world_transform[3];

		culling_data->view  = camera.view;
		culling_data->P00   = camera.projection[0][0];
		culling_data->P11   = camera.projection[1][1];
		culling_data->znear = camera.near_plane;
		culling_data->zfar  = camera.far_plane;

		camera_data->position             = transform.world_transform[3];
		camera_data->last_view_projection = camera.last_view_projection;
		camera_data->view_inverse         = transform.world_transform;
		camera_data->projection_inverse   = glm::inverse(camera.projection);
		camera_data->frame_num            = camera.frame_count;

		// Jitter camera
		auto jitter_offset       = m_halton_sequence[(m_frame_count++) % 16];
		camera_data->prev_jitter = camera_data->jitter;
		camera_data->jitter.x    = (2.f * jitter_offset[0] - 1.f) / static_cast<float>(Renderer::instance()->getRenderTargetExtent().width);
		camera_data->jitter.y    = (2.f * jitter_offset[1] - 1.f) / static_cast<float>(Renderer::instance()->getRenderTargetExtent().height);
		glm::mat4 projection     = camera.projection;
		projection[2][0] += camera_data->jitter.x;
		projection[2][1] += camera_data->jitter.y;
		camera_data->view_projection = projection * camera.view;

		for (size_t i = 0; i < 6; i++)
		{
			camera_data->frustum[i] = glm::vec4(camera.frustum.planes[i].normal, camera.frustum.planes[i].constant);
		}

		camera.last_view_projection = camera.view_projection;
		camera.frame_count++;
	}

	culling_data->meshlet_count  = Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count;
	culling_data->instance_count = Renderer::instance()->Render_Stats.static_mesh_count.instance_count;
	culling_data->zbuffer_width  = static_cast<float>(Renderer::instance()->getViewportExtent().width);
	culling_data->zbuffer_height = static_cast<float>(Renderer::instance()->getViewportExtent().height);

	Renderer::instance()->Render_Buffer.Camera_Buffer.unmap();
	Renderer::instance()->Render_Buffer.Culling_Buffer.unmap();

	GraphicsContext::instance()->getProfiler().endSample("Camera Update");
}
}        // namespace Ilum::sym
