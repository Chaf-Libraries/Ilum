#include "LightUpdate.hpp"

#include "Scene/Component/Camera.hpp"
#include "Scene/Component/Light.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Profiler.hpp"

#include "Renderer/Renderer.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <tbb/tbb.h>

namespace Ilum::sym
{
void LightUpdate::run()
{
	GraphicsContext::instance()->getProfiler().beginSample("Light Update");

	// Collect light infos
	auto directional_lights = Scene::instance()->getRegistry().group<cmpt::DirectionalLight>(entt::get<cmpt::Tag, cmpt::Transform>);
	auto point_lights       = Scene::instance()->getRegistry().group<cmpt::PointLight>(entt::get<cmpt::Tag, cmpt::Transform>);
	auto spot_lights        = Scene::instance()->getRegistry().group<cmpt::SpotLight>(entt::get<cmpt::Tag, cmpt::Transform>);

	// Enlarge buffer
	if (Renderer::instance()->Render_Buffer.Directional_Light_Buffer.getSize() / sizeof(cmpt::SpotLight) < directional_lights.size())
	{
		GraphicsContext::instance()->getQueueSystem().waitAll();
		Renderer::instance()->Render_Buffer.Directional_Light_Buffer = Buffer(directional_lights.size() * sizeof(cmpt::DirectionalLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		Renderer::instance()->update();
	}
	if (Renderer::instance()->Render_Buffer.Spot_Light_Buffer.getSize() / sizeof(cmpt::SpotLight) < spot_lights.size())
	{
		GraphicsContext::instance()->getQueueSystem().waitAll();
		Renderer::instance()->Render_Buffer.Spot_Light_Buffer = Buffer(spot_lights.size() * sizeof(cmpt::SpotLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		Renderer::instance()->update();
	}
	if (Renderer::instance()->Render_Buffer.Point_Light_Buffer.getSize() / sizeof(cmpt::PointLight) < point_lights.size())
	{
		GraphicsContext::instance()->getQueueSystem().waitAll();
		Renderer::instance()->Render_Buffer.Point_Light_Buffer = Buffer(point_lights.size() * sizeof(cmpt::PointLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		Renderer::instance()->update();
	}

	// Copy Buffer
	Renderer::instance()->Render_Stats.light_count.directional_light_count = 0;
	directional_lights.each([](entt::entity entity, cmpt::DirectionalLight &light, cmpt::Tag &tag, cmpt::Transform &transform) {
		light.position = transform.world_transform[3];
		// Update Cascade
		auto &camera_entity = Renderer::instance()->Main_Camera;
		if (camera_entity && (camera_entity.hasComponent<cmpt::PerspectiveCamera>() || camera_entity.hasComponent<cmpt::OrthographicCamera>()))
		{
			const cmpt::Camera *camera = camera_entity.hasComponent<cmpt::PerspectiveCamera>() ? static_cast<cmpt::Camera *>(&camera_entity.getComponent<cmpt::PerspectiveCamera>()) : static_cast<cmpt::Camera *>(&camera_entity.getComponent<cmpt::OrthographicCamera>());

			float cascade_splits[4] = {0.f};

			float near_clip  = camera->near_plane;
			float far_clip   = camera->far_plane;
			float clip_range = far_clip - near_clip;
			float ratio      = far_clip / near_clip;

			// Calculate split depths based on view camera frustum
			for (uint32_t i = 0; i < 4; i++)
			{
				float p           = (static_cast<float>(i) + 1.f) / 4.f;
				float log         = near_clip * std::pow(ratio, p);
				float uniform     = near_clip + clip_range * p;
				float d           = 0.95f * (log - uniform) + uniform;
				cascade_splits[i] = (d - near_clip) / clip_range;
			}

			// Calculate orthographic projection matrix for each cascade
			float last_split_dist = 0.f;
			for (uint32_t i = 0; i < 4; i++)
			{
				float split_dist = cascade_splits[i];

				glm::vec3 frustum_corners[8] = {
				    glm::vec3(-1.f, -1.f, 0.f),
				    glm::vec3(1.f, -1.f, 0.f),
				    glm::vec3(1.f, 1.f, 0.f),
				    glm::vec3(-1.f, 1.f, 0.f),
				    glm::vec3(-1.f, -1.f, 1.f),
				    glm::vec3(1.f, -1.f, 1.f),
				    glm::vec3(1.f, 1.f, 1.f),
				    glm::vec3(-1.f, 1.f, 1.f),
				};

				// Project frustum corners into world space
				glm::mat4 inv_cam = glm::inverse(camera->view_projection);
				for (uint32_t i = 0; i < 8; i++)
				{
					glm::vec4 inv_corner = inv_cam * glm::vec4(frustum_corners[i], 1.f);
					frustum_corners[i]   = glm::vec3(inv_corner / inv_corner.w);
				}

				for (uint32_t i = 0; i < 4; i++)
				{
					glm::vec3 corner_ray   = frustum_corners[i + 4] - frustum_corners[i];
					frustum_corners[i + 4] = frustum_corners[i] + corner_ray * split_dist;
					frustum_corners[i]     = frustum_corners[i] + corner_ray * last_split_dist;
				}

				// Get frustum center
				glm::vec3 frustum_center = glm::vec3(0.0f);
				for (uint32_t i = 0; i < 8; i++)
				{
					frustum_center += frustum_corners[i];
				}
				frustum_center /= 8.0f;

				float radius = 0.0f;
				for (uint32_t i = 0; i < 8; i++)
				{
					float distance = glm::length(frustum_corners[i] - frustum_center);
					radius         = glm::max(radius, distance);
				}
				radius = std::ceil(radius * 16.0f) / 16.0f;

				glm::vec3 max_extents = glm::vec3(radius);
				glm::vec3 min_extents = -max_extents;

				glm::vec3 light_dir = glm::normalize(-light.position);

				glm::mat4 light_view_matrix  = glm::lookAt(frustum_center - light_dir * max_extents.z, frustum_center, glm::vec3(0.0f, 1.0f, 0.0f));
				glm::mat4 light_ortho_matrix = glm::ortho(min_extents.x, max_extents.x, min_extents.y, max_extents.y, 0.0f, max_extents.z - min_extents.z);

				// Store split distance and matrix in cascade
				light.split_depth[i]     = -(near_clip + split_dist * clip_range);
				light.view_projection[i] = light_ortho_matrix * light_view_matrix;

				last_split_dist = cascade_splits[i];
			}
		}
		std::memcpy(reinterpret_cast<cmpt::DirectionalLight *>(Renderer::instance()->Render_Buffer.Directional_Light_Buffer.map()) + Renderer::instance()->Render_Stats.light_count.directional_light_count++,
		            &light, sizeof(cmpt::DirectionalLight));
	});

	Renderer::instance()->Render_Stats.light_count.point_light_count = 0;
	point_lights.each([](entt::entity entity, cmpt::PointLight &light, cmpt::Tag &tag, cmpt::Transform &transform) {
		light.position = transform.world_transform[3];
		std::memcpy(reinterpret_cast<cmpt::PointLight *>(Renderer::instance()->Render_Buffer.Point_Light_Buffer.map()) + Renderer::instance()->Render_Stats.light_count.point_light_count++,
		            &light, sizeof(cmpt::PointLight));
	});

	Renderer::instance()->Render_Stats.light_count.spot_light_count = 0;
	spot_lights.each([](entt::entity entity, cmpt::SpotLight &light, cmpt::Tag &tag, cmpt::Transform &transform) {
		light.position        = transform.world_transform[3];
		light.view_projection = glm::perspective(2.f * glm::acos(light.outer_cut_off), 1.0f, 0.01f, 1000.f) * glm::lookAt(transform.translation, transform.translation + light.direction, glm::vec3(0.f, 1.f, 0.f));
		std::memcpy(reinterpret_cast<cmpt::SpotLight *>(Renderer::instance()->Render_Buffer.Spot_Light_Buffer.map()) + Renderer::instance()->Render_Stats.light_count.spot_light_count++,
		            &light, sizeof(cmpt::SpotLight));
	});

	Renderer::instance()->Render_Buffer.Directional_Light_Buffer.unmap();
	Renderer::instance()->Render_Buffer.Spot_Light_Buffer.unmap();
	Renderer::instance()->Render_Buffer.Point_Light_Buffer.unmap();

	GraphicsContext::instance()->getProfiler().endSample("Light Update");
}
}        // namespace Ilum::sym