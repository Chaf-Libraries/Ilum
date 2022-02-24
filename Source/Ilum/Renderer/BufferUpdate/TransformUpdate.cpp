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

		if (instance_count * sizeof(PerInstanceData) > Renderer::instance()->Render_Buffer.Instance_Buffer.getSize())
		{
			GraphicsContext::instance()->getQueueSystem().waitAll();
			Renderer::instance()->Render_Buffer.Instance_Buffer            = Buffer(instance_count * sizeof(PerInstanceData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			Renderer::instance()->Render_Buffer.VKTransfrom_Buffer         = Buffer(instance_count * sizeof(VkTransformMatrixKHR), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_CPU_TO_GPU);
			Renderer::instance()->Render_Buffer.Instance_Visibility_Buffer = Buffer(instance_count * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
			Renderer::instance()->update();
		}

		PerInstanceData *     instance_data         = reinterpret_cast<PerInstanceData *>(Renderer::instance()->Render_Buffer.Instance_Buffer.map());
		VkTransformMatrixKHR *transform_matrix_data = reinterpret_cast<VkTransformMatrixKHR *>(Renderer::instance()->Render_Buffer.VKTransfrom_Buffer.map());

		std::vector<VkAccelerationStructureBuildRangeInfoKHR> range_infos(instance_count);

		tbb::parallel_for(tbb::blocked_range<size_t>(0, meshlet_view.size()), [&meshlet_view, &instance_data, &transform_matrix_data, &range_infos, instance_offset](const tbb::blocked_range<size_t> &r) {
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

					range_infos[instance_idx].firstVertex     = instance.vertex_offset;
					range_infos[instance_idx].primitiveCount  = instance.index_count / 3;
					range_infos[instance_idx].primitiveOffset = instance.index_offset / 3;
					range_infos[instance_idx].transformOffset = instance_idx;

					transform_matrix_data[instance_idx] = VkTransformMatrixKHR{
					    instance.transform[0][0], instance.transform[0][1], instance.transform[0][2], instance.transform[0][3],
					    instance.transform[1][0], instance.transform[1][1], instance.transform[1][2], instance.transform[1][3],
					    instance.transform[2][0], instance.transform[2][1], instance.transform[2][2], instance.transform[2][3]};
				}
			}
		});
		Renderer::instance()->Render_Buffer.Instance_Buffer.unmap();
		Renderer::instance()->Render_Buffer.VKTransfrom_Buffer.unmap();

		// Update BLAS
		auto &blas = Renderer::instance()->Render_Buffer.Bottom_Level_AS;
		blas.reset();

		VkAccelerationStructureGeometryTrianglesDataKHR triangle_data = {};
		triangle_data.sType                                           = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
		triangle_data.vertexFormat                                    = VK_FORMAT_R32G32B32_SFLOAT;        // vec3 vertex position data.
		triangle_data.vertexData.deviceAddress                        = Renderer::instance()->Render_Buffer.Static_Vertex_Buffer.getDeviceAddress();
		triangle_data.vertexStride                                    = sizeof(Vertex);
		triangle_data.indexType                                       = VK_INDEX_TYPE_UINT32;
		triangle_data.indexData.deviceAddress                         = Renderer::instance()->Render_Buffer.Static_Index_Buffer.getDeviceAddress();
		triangle_data.transformData.hostAddress                       = nullptr;
		triangle_data.transformData.deviceAddress                     = 0;
		triangle_data.maxVertex                                       = Renderer::instance()->getResourceCache().getVerticesCount();

		VkAccelerationStructureBuildRangeInfoKHR range_info = {};
		range_info.firstVertex                              = 0;
		range_info.primitiveOffset                          = 0;
		range_info.transformOffset                          = 0;
		range_info.primitiveCount                           = Renderer::instance()->getResourceCache().getIndicesCount() / 3;

		//for (auto& range_info : range_infos)
		//{
		blas.add(triangle_data);
		blas.add(range_info);
		//}

		blas.build();
	}

	cmpt::Transform::update = false;
	GraphicsContext::instance()->getProfiler().endSample("Transform Update");
}
}        // namespace Ilum::sym
