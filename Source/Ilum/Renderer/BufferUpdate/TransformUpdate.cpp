#include "TransformUpdate.hpp"

#include "Scene/Component/Hierarchy.hpp"
#include "Scene/Component/Renderable.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Profiler.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <tbb/tbb.h>

namespace Ilum::sym
{
inline void transform_recrusive(entt::entity entity)
{
	if (entity == entt::null)
	{
		return;
	}

	auto &transform = Entity(entity).getComponent<cmpt::Transform>();
	auto &hierarchy = Entity(entity).getComponent<cmpt::Hierarchy>();
	if (hierarchy.parent != entt::null)
	{
		transform.local_transform = glm::scale(glm::translate(glm::mat4(1.f), transform.translation) * glm::mat4_cast(glm::qua<float>(glm::radians(transform.rotation))), transform.scale);
		transform.world_transform = Entity(hierarchy.parent).getComponent<cmpt::Transform>().world_transform * transform.local_transform;
	}

	auto child = hierarchy.first;

	while (child != entt::null)
	{
		transform_recrusive(child);
		child = Entity(child).getComponent<cmpt::Hierarchy>().next;
	}
}

void TransformUpdate::run()
{
	GraphicsContext::instance()->getProfiler().beginSample("Transform Update");

	if (cmpt::Transform::update)
	{
		auto group = Scene::instance()->getRegistry().group<>(entt::get<cmpt::Transform, cmpt::Hierarchy>);

		std::vector<entt::entity> roots;

		// Find roots
		for (auto &entity : group)
		{
			auto &[hierarchy, transform] = group.get<cmpt::Hierarchy, cmpt::Transform>(entity);

			if (hierarchy.parent == entt::null)
			{
				transform.local_transform = glm::scale(glm::translate(glm::mat4(1.f), transform.translation) * glm::mat4_cast(glm::qua<float>(glm::radians(transform.rotation))), transform.scale);
				transform.world_transform = transform.local_transform;
				roots.emplace_back(entity);
			}
		}

		tbb::parallel_for_each(roots.begin(), roots.end(), [&group](auto entity) {
			transform_recrusive(entity);
		});
	}

	if (cmpt::Transform::update || cmpt::MeshletRenderer::update)
	{
		m_motionless_count = 0;
	}

	// Update all transform and update last transform for another time
	if (m_motionless_count <= 1)
	{
		m_motionless_count++;

		// Update static mesh transform buffer
		auto meshlet_view = Scene::instance()->getRegistry().view<cmpt::MeshletRenderer>();

		// Collect instance data
		std::vector<size_t> instance_offset(meshlet_view.size());

		size_t instance_count = 0;

		for (size_t i = 0; i < meshlet_view.size(); i++)
		{
			instance_offset[i] += instance_count;

			auto &meshlet_renderer = Entity(meshlet_view[i]).getComponent<cmpt::MeshletRenderer>();

			if (Renderer::instance()->getResourceCache().hasModel(meshlet_renderer.model))
			{
				auto model = Renderer::instance()->getResourceCache().loadModel(meshlet_renderer.model);
				instance_count += model.get().submeshes.size();
			}
		}

		if (instance_count * sizeof(PerInstanceData) > Renderer::instance()->Render_Buffer.Instance_Buffer.GetSize())
		{
			Graphics::RenderContext::WaitDevice();
			Renderer::instance()->Render_Buffer.Instance_Buffer            = Graphics::Buffer(Graphics::RenderContext::GetDevice(), instance_count * sizeof(PerInstanceData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			Renderer::instance()->Render_Buffer.Instance_Visibility_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice(), instance_count * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
			Renderer::instance()->update();
		}

		PerInstanceData *instance_data = reinterpret_cast<PerInstanceData *>(Renderer::instance()->Render_Buffer.Instance_Buffer.Map());

		tbb::parallel_for(tbb::blocked_range<size_t>(0, meshlet_view.size()), [&meshlet_view, &instance_data, instance_offset](const tbb::blocked_range<size_t> &r) {
			for (size_t i = r.begin(); i != r.end(); i++)
			{
				auto        entity           = Entity(meshlet_view[i]);
				const auto &meshlet_renderer = entity.getComponent<cmpt::MeshletRenderer>();
				const auto &transform        = entity.getComponent<cmpt::Transform>();

				if (!Renderer::instance()->getResourceCache().hasModel(meshlet_renderer.model))
				{
					continue;
				}

				auto model = Renderer::instance()->getResourceCache().loadModel(meshlet_renderer.model);

				uint32_t submesh_index = 0;
				for (auto &submesh : model.get().submeshes)
				{
					auto &instance                = instance_data[instance_offset[i] + submesh_index++];
					instance.entity_id            = static_cast<uint32_t>(meshlet_view[i]);
					instance.bbox_min             = submesh.bounding_box.min_;
					instance.bbox_max             = submesh.bounding_box.max_;
					instance.pre_transform        = submesh.pre_transform;
					instance.last_world_transform = instance.world_transform;
					instance.world_transform      = transform.world_transform;
				}
			}
		});
		Renderer::instance()->Render_Buffer.Instance_Buffer.Unmap();
	}

	cmpt::Transform::update = false;
	GraphicsContext::instance()->getProfiler().endSample("Transform Update");
}
}        // namespace Ilum::sym
