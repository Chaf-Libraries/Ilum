#include "RenderQueue.hpp"

#include "Scene/Component/MeshRenderer.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "File/FileSystem.hpp"

#include "tbb/tbb.h"

namespace Ilum
{
bool RenderQueue::update()
{
	// Update culling data
	auto *culling_data = reinterpret_cast<CullingData *>(Culling_Buffer.map());
	culling_data->last_view = culling_data->view;
	culling_data->view             = Renderer::instance()->Main_Camera.view;
	culling_data->P00              = Renderer::instance()->Main_Camera.projection[0][0];
	culling_data->P11              = Renderer::instance()->Main_Camera.projection[1][1];
	culling_data->znear            = Renderer::instance()->Main_Camera.near_plane;
	culling_data->zfar             = Renderer::instance()->Main_Camera.far_plane;
	culling_data->meshlet_count    = Renderer::instance()->Meshlet_Count;
	culling_data->instance_count    = Renderer::instance()->Instance_Count;
	culling_data->frustum_enable   = Renderer::instance()->Culling.frustum_culling;
	culling_data->backface_enable  = Renderer::instance()->Culling.backface_culling;
	culling_data->occlusion_enable = Renderer::instance()->Culling.occulsion_culling;
	culling_data->zbuffer_width    = static_cast<float>(Renderer::instance()->Last_Frame.hiz_buffer->getWidth());
	culling_data->zbuffer_height   = static_cast<float>(Renderer::instance()->Last_Frame.hiz_buffer->getHeight());
	Culling_Buffer.unmap();

	// Check for update
	bool                  update         = false;
	std::atomic<uint32_t> instance_count = 0;

	auto group = Scene::instance()->getRegistry().group<cmpt::MeshRenderer, cmpt::Tag, cmpt::Transform>();
	tbb::parallel_for_each(group.begin(), group.end(), [&instance_count, this](auto entity) {
		auto &mesh_renderer = Entity(entity).getComponent<cmpt::MeshRenderer>();
		if (!Renderer::instance()->getResourceCache().hasModel(mesh_renderer.model))
		{
			return;
		}

		auto &model = Renderer::instance()->getResourceCache().loadModel(mesh_renderer.model);

		tbb::parallel_for_each(mesh_renderer.materials.begin(), mesh_renderer.materials.end(), [&instance_count](scope<IMaterial> &material) {
			instance_count.fetch_add(1, std::memory_order_relaxed);
		});
	});

	if (instance_count != Renderer::instance()->Instance_Count || ResourceCache::update || cmpt::MeshRenderer::update || cmpt::Tag::update)
	{
		cmpt::MeshRenderer::update = false;
		cmpt::Tag::update          = false;

		Renderer::instance()->Indices_Count  = 0;
		Renderer::instance()->Instance_Count = 0;
		Renderer::instance()->Meshlet_Count  = 0;

		std::vector<PerInstanceData> instance_data;
		std::vector<PerMeshletData>  meshlet_data;

		const auto group = Scene::instance()->getRegistry().group<>(entt::get<cmpt::MeshRenderer, cmpt::Tag, cmpt::Transform>);
		group.each([&](const entt::entity &entity, const cmpt::MeshRenderer &mesh_renderer, const cmpt::Tag &tag, const cmpt::Transform &transform) {
			if (!tag.active || !Renderer::instance()->getResourceCache().hasModel(mesh_renderer.model))
			{
				return;
			}

			auto &model = Renderer::instance()->getResourceCache().loadModel(mesh_renderer.model);

			for (uint32_t i = 0; i < mesh_renderer.materials.size(); i++)
			{
				auto &submesh      = model.get().submeshes[i];
				auto &material_ptr = mesh_renderer.materials[i];

				auto *material = material_ptr->type() == typeid(material::DisneyPBR) ? static_cast<material::DisneyPBR *>(material_ptr.get()) : nullptr;

				for (uint32_t j = submesh.meshlet_offset; j < submesh.meshlet_offset + submesh.meshlet_count; j++)
				{
					const auto &meshlet = model.get().meshlets[j];

					Renderer::instance()->Meshlet_Count++;

					Renderer::instance()->Indices_Count += meshlet.indices_count;

					PerMeshletData meshlet_;
					// Vertex&Index info
					meshlet_.vertex_offset = model.get().vertices_offset + meshlet.vertices_offset;
					meshlet_.index_offset  = model.get().indices_offset + meshlet.indices_offset;
					meshlet_.index_count   = meshlet.indices_count;
					// Meshlet bounds
					std::memcpy(&meshlet_.center, meshlet.bounds.center, 3 * sizeof(float));
					meshlet_.radius = meshlet.bounds.radius;
					std::memcpy(&meshlet_.cone_apex, meshlet.bounds.cone_apex, 3 * sizeof(float));
					meshlet_.cone_cutoff = meshlet.bounds.cone_cutoff;
					std::memcpy(&meshlet_.cone_axis, meshlet.bounds.cone_axis, 3 * sizeof(float));

					meshlet_.instance_id = Renderer::instance()->Instance_Count;

					meshlet_data.push_back(meshlet_);
				}

				PerInstanceData instance;
				// Instance Transform
				instance.world_transform = transform.world_transform;
				instance.pre_transform   = submesh.pre_transform;
				// Instance BoundingBox
				instance.min_ = submesh.bounding_box.min_;
				instance.max_ = submesh.bounding_box.max_;
				// Instance Material
				if (material)
				{
					instance.base_color          = material->base_color;
					instance.emissive_color      = material->emissive_color;
					instance.metallic_factor     = material->metallic_factor;
					instance.roughness_factor    = material->roughness_factor;
					instance.emissive_intensity  = material->emissive_intensity;
					instance.albedo_map          = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->albedo_map));
					instance.normal_map          = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->normal_map));
					instance.metallic_map        = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->metallic_map));
					instance.roughness_map       = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->roughness_map));
					instance.emissive_map        = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->emissive_map));
					instance.ao_map              = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->ao_map));
					instance.displacement_height = material->displacement_height;
					instance.displacement_map    = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->displacement_map));
				}
				instance.entity_id = static_cast<uint32_t>(entity);
				instance_data.push_back(instance);

				Renderer::instance()->Instance_Count++;
			}
		});

		// Enlarge buffer
		if (Instance_Buffer.getSize() < instance_data.size() * sizeof(PerInstanceData))
		{
			GraphicsContext::instance()->getQueueSystem().waitAll();
			Instance_Buffer = Buffer(static_cast<VkDeviceSize>(static_cast<double>(instance_data.size() * sizeof(PerInstanceData)) * 1.1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			Instance_Visibility_Buffer = Buffer(static_cast<VkDeviceSize>(static_cast<double>(instance_data.size() * sizeof(uint32_t)) * 1.1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

			update = true;
		}
		if (Meshlet_Buffer.getSize() < meshlet_data.size() * sizeof(PerMeshletData))
		{
			GraphicsContext::instance()->getQueueSystem().waitAll();
			Meshlet_Buffer = Buffer(static_cast<VkDeviceSize>(static_cast<double>(meshlet_data.size() * sizeof(PerMeshletData)) * 1.1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			Command_Buffer = Buffer(static_cast<VkDeviceSize>(static_cast<double>(meshlet_data.size() * sizeof(VkDrawIndexedIndirectCommand)) * 1.1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
			Draw_Buffer    = Buffer(static_cast<VkDeviceSize>(static_cast<double>(meshlet_data.size() * sizeof(uint32_t)) * 1.1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

			update = true;
		}

		// Copy buffer
		std::memcpy(Instance_Buffer.map(), instance_data.data(), instance_data.size() * sizeof(PerInstanceData));
		std::memcpy(Meshlet_Buffer.map(), meshlet_data.data(), meshlet_data.size() * sizeof(PerMeshletData));
		Instance_Buffer.unmap();
		Meshlet_Buffer.unmap();
	}
	else if (cmpt::Transform::update)
	{
		// Update transform only
		std::vector<PerInstanceData> instance_data(Renderer::instance()->Instance_Count);
		std::memcpy(instance_data.data(), Instance_Buffer.map(), Renderer::instance()->Instance_Count * sizeof(PerInstanceData));

		uint32_t   idx   = 0;
		const auto group = Scene::instance()->getRegistry().group<>(entt::get<cmpt::MeshRenderer, cmpt::Tag, cmpt::Transform>);
		group.each([&](const entt::entity &entity, const cmpt::MeshRenderer &mesh_renderer, const cmpt::Tag &tag, const cmpt::Transform &transform) {
			if (!Renderer::instance()->getResourceCache().hasModel(mesh_renderer.model))
			{
				return;
			}

			auto &model = Renderer::instance()->getResourceCache().loadModel(mesh_renderer.model);
			for (uint32_t i = 0; i < mesh_renderer.materials.size(); i++)
			{
				auto &item           = instance_data[idx++];
				item.world_transform = transform.world_transform;
			}
		});

		//	// Copy buffer
		std::memcpy(Instance_Buffer.map(), instance_data.data(), Renderer::instance()->Instance_Count * sizeof(PerInstanceData));

		Instance_Buffer.unmap();
	}

	return update;
}
}        // namespace Ilum