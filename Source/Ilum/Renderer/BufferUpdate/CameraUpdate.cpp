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
// Camera jitter
inline float halton_sequence(uint32_t base, uint32_t index)
{
	float result = 0.f;
	float f      = 1.f;

	while (index > 0)
	{
		f /= static_cast<float>(base);
		result += f * (index % base);
		index = static_cast<uint32_t>(floorf(static_cast<float>(index) / static_cast<float>(base)));
	}

	return result;
}

CameraUpdate::CameraUpdate()
{
	for (uint32_t i = 1; i <= HALTION_SAMPLES; i++)
	{
		m_jitter_samples.push_back(glm::vec2(2.f * halton_sequence(2, i) - 1.f, 2.f * halton_sequence(3, i) - 1.f));
	}
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
			auto extent                                                  = Renderer::instance()->getRenderTargetExtent();
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

	// Update camera jitter
	if (Renderer::instance()->TAA.enable)
	{
		Renderer::instance()->TAA.prev_jitter = Renderer::instance()->TAA.current_jitter;
		uint32_t sample_idx = static_cast<uint32_t>(GraphicsContext::instance()->getFrameCount() % m_jitter_samples.size());
		glm::vec2 halton     = m_jitter_samples[sample_idx];

		auto rt_extent   = Renderer::instance()->getRenderTargetExtent();
		
		Renderer::instance()->TAA.current_jitter = glm::vec2(halton.x / static_cast<float>(rt_extent.width), halton.y / static_cast<float>(rt_extent.height));
	}
	else
	{
		Renderer::instance()->TAA.prev_jitter    = glm::vec2(0.f);
		Renderer::instance()->TAA.current_jitter = glm::vec2(0.f);
	}

	const auto &transform = camera_entity.getComponent<cmpt::Transform>();

	CameraData * camera_data          = reinterpret_cast<CameraData *>(Renderer::instance()->Render_Buffer.Camera_Buffer.Map());
	CullingData *culling_data         = reinterpret_cast<CullingData *>(Renderer::instance()->Render_Buffer.Culling_Buffer.Map());
	culling_data->last_view           = culling_data->view;

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

		camera_data->position        = transform.world_transform[3];
		camera_data->view_projection = glm::translate(glm::mat4(1.f), glm::vec3(Renderer::instance()->TAA.current_jitter, 0.f)) * camera.view_projection;
		camera_data->last_view_projection = glm::translate(glm::mat4(1.f), glm::vec3(Renderer::instance()->TAA.current_jitter, 0.f)) * camera.last_view_projection;

		for (size_t i = 0; i < 6; i++)
		{
			camera_data->frustum[i] = glm::vec4(camera.frustum.planes[i].normal, camera.frustum.planes[i].constant);
		}

		camera.last_view_projection = camera.view_projection;
	}
	else if (camera_entity.hasComponent<cmpt::OrthographicCamera>())
	{
		auto &camera           = camera_entity.getComponent<cmpt::OrthographicCamera>();
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

		camera_data->position        = transform.world_transform[3];
		camera_data->view_projection = glm::translate(glm::mat4(1.f), glm::vec3(Renderer::instance()->TAA.current_jitter, 0.f)) * camera.view_projection;
		camera_data->last_view_projection = glm::translate(glm::mat4(1.f), glm::vec3(Renderer::instance()->TAA.current_jitter, 0.f)) * camera.last_view_projection;

		for (size_t i = 0; i < 6; i++)
		{
			camera_data->frustum[i] = glm::vec4(camera.frustum.planes[i].normal, camera.frustum.planes[i].constant);
		}

		camera.last_view_projection = camera.view_projection;
	}

	culling_data->meshlet_count    = Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count;
	culling_data->instance_count   = Renderer::instance()->Render_Stats.static_mesh_count.instance_count;
	culling_data->frustum_enable   = Renderer::instance()->Culling.frustum_culling;
	culling_data->backface_enable  = Renderer::instance()->Culling.backface_culling;
	culling_data->occlusion_enable = Renderer::instance()->Culling.occulsion_culling;
	culling_data->zbuffer_width    = static_cast<float>(Renderer::instance()->Last_Frame.hiz_buffer->GetWidth());
	culling_data->zbuffer_height   = static_cast<float>(Renderer::instance()->Last_Frame.hiz_buffer->GetHeight());

	Renderer::instance()->Render_Buffer.Camera_Buffer.Unmap();
	Renderer::instance()->Render_Buffer.Culling_Buffer.Unmap();

	GraphicsContext::instance()->getProfiler().endSample("Camera Update");
}
}        // namespace Ilum::sym
