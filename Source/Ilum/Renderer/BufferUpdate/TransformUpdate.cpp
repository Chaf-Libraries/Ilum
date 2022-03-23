#include "TransformUpdate.hpp"

#include "Scene/Component/Camera.hpp"
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
#include <glm/gtc/type_ptr.hpp>

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

	auto group = Scene::instance()->getRegistry().group<>(entt::get<cmpt::Transform, cmpt::Hierarchy>);

	if (group.empty())
	{
		return;
	}

	if (cmpt::Transform::update)
	{
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

	if (cmpt::Transform::update || cmpt::StaticMeshRenderer::update)
	{
		m_motionless_count = 0;
	}

	// Update all transform and update last transform for another time
	if (m_motionless_count <= 1)
	{
		m_motionless_count++;

		// Update static mesh transform buffer
		auto meshlet_view = Scene::instance()->getRegistry().view<cmpt::StaticMeshRenderer>();

		// Collect instance data
		std::vector<size_t> instance_offset(meshlet_view.size());

		size_t instance_count = 0;

		for (size_t i = 0; i < meshlet_view.size(); i++)
		{
			instance_offset[i] += instance_count;

			auto &meshlet_renderer = Entity(meshlet_view[i]).getComponent<cmpt::StaticMeshRenderer>();

			if (Renderer::instance()->getResourceCache().hasModel(meshlet_renderer.model))
			{
				auto model = Renderer::instance()->getResourceCache().loadModel(meshlet_renderer.model);
				instance_count += model.get().submeshes.size();
			}
		}

		if (instance_count * sizeof(PerInstanceData) > Renderer::instance()->Render_Buffer.Instance_Buffer.getSize())
		{
			GraphicsContext::instance()->getQueueSystem().waitAll();
			Renderer::instance()->Render_Buffer.Instance_Buffer            = Buffer(instance_count * sizeof(PerInstanceData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			Renderer::instance()->Render_Buffer.Instance_Visibility_Buffer = Buffer(instance_count * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
			Renderer::instance()->Render_Buffer.RTXInstance_Buffer         = Buffer(instance_count * sizeof(VkAccelerationStructureInstanceKHR), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_CPU_TO_GPU);
			Renderer::instance()->update();
		}

		PerInstanceData *                   instance_data    = reinterpret_cast<PerInstanceData *>(Renderer::instance()->Render_Buffer.Instance_Buffer.map());
		VkAccelerationStructureInstanceKHR *as_instance_data = reinterpret_cast<VkAccelerationStructureInstanceKHR *>(Renderer::instance()->Render_Buffer.RTXInstance_Buffer.map());

		tbb::parallel_for(tbb::blocked_range<size_t>(0, meshlet_view.size()), [&meshlet_view, &instance_data, &as_instance_data, instance_offset](const tbb::blocked_range<size_t> &r) {
			for (size_t i = r.begin(); i != r.end(); i++)
			{
				auto        entity           = Entity(meshlet_view[i]);
				const auto &meshlet_renderer = entity.getComponent<cmpt::StaticMeshRenderer>();
				const auto &transform        = entity.getComponent<cmpt::Transform>();

				if (!Renderer::instance()->getResourceCache().hasModel(meshlet_renderer.model))
				{
					continue;
				}

				auto model = Renderer::instance()->getResourceCache().loadModel(meshlet_renderer.model);

				uint32_t submesh_index = 0;
				for (auto &submesh : model.get().submeshes)
				{
					size_t instance_idx = instance_offset[i] + submesh_index++;

					auto &instance          = instance_data[instance_idx];
					instance.entity_id      = static_cast<uint32_t>(meshlet_view[i]);
					instance.bbox_min       = submesh.bounding_box.min_;
					instance.bbox_max       = submesh.bounding_box.max_;
					instance.last_transform = instance.transform;
					instance.transform      = transform.world_transform * submesh.pre_transform;
					instance.vertex_offset  = model.get().vertices_offset + submesh.vertices_offset;
					instance.index_offset   = model.get().indices_offset + submesh.indices_offset;
					instance.index_count    = submesh.indices_count;

					auto transform = glm::mat3x4(glm::transpose(instance.transform));
					std::memcpy(&as_instance_data[instance_idx].transform, &transform, sizeof(VkTransformMatrixKHR));
					as_instance_data[instance_idx].instanceCustomIndex                    = 0;
					as_instance_data[instance_idx].mask                                   = 0xFF;
					as_instance_data[instance_idx].instanceShaderBindingTableRecordOffset = 0;
					as_instance_data[instance_idx].flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
					as_instance_data[instance_idx].accelerationStructureReference         = submesh.bottom_level_as.getDeviceAddress();
				}
			}
		});
		Renderer::instance()->Render_Buffer.Instance_Buffer.unmap();
		Renderer::instance()->Render_Buffer.RTXInstance_Buffer.unmap();

		VkAccelerationStructureGeometryKHR geometry_info    = {};
		geometry_info.sType                                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		geometry_info.geometryType                          = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		geometry_info.flags                                 = VK_GEOMETRY_OPAQUE_BIT_KHR;
		geometry_info.geometry.instances.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
		geometry_info.geometry.instances.arrayOfPointers    = VK_FALSE;
		geometry_info.geometry.instances.data.deviceAddress = Renderer::instance()->Render_Buffer.RTXInstance_Buffer.getDeviceAddress();

		VkAccelerationStructureBuildRangeInfoKHR range_info = {};
		range_info.primitiveCount                           = static_cast<uint32_t>(instance_count);
		range_info.primitiveOffset                          = 0;
		range_info.firstVertex                              = 0;
		range_info.transformOffset                          = 0;

		if (Renderer::instance()->Render_Buffer.Top_Level_AS.build(geometry_info, range_info))
		{
			Renderer::instance()->rebuild();
		}

		auto &camera_entity = Renderer::instance()->Main_Camera;

		if (camera_entity && (camera_entity.hasComponent<cmpt::PerspectiveCamera>() || camera_entity.hasComponent<cmpt::OrthographicCamera>()))
		{
			cmpt::Camera *camera = camera_entity.hasComponent<cmpt::PerspectiveCamera>() ?
                                       static_cast<cmpt::Camera *>(&camera_entity.getComponent<cmpt::PerspectiveCamera>()) :
                                       static_cast<cmpt::Camera *>(&camera_entity.getComponent<cmpt::OrthographicCamera>());

			camera->frame_count = 0;
		}
	}

	cmpt::Transform::update = false;
	GraphicsContext::instance()->getProfiler().endSample("Transform Update");
}
}        // namespace Ilum::sym
