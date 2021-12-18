#include "RenderQueue.hpp"

#include "Scene/Component/Camera.hpp"
#include "Scene/Component/Renderable.hpp"
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
	if (Renderer::instance()->hasMainCamera())
	{
		cmpt::Camera *main_camera = Renderer::instance()->Main_Camera.hasComponent<cmpt::PerspectiveCamera>() ?
                                        static_cast<cmpt::Camera *>(&Renderer::instance()->Main_Camera.getComponent<cmpt::PerspectiveCamera>()) :
                                        static_cast<cmpt::Camera *>(&Renderer::instance()->Main_Camera.getComponent<cmpt::OrthographicCamera>());

		// Update culling data
		auto *culling_data             = reinterpret_cast<CullingData *>(Culling_Buffer.map());
		culling_data->last_view        = culling_data->view;
		culling_data->view             = main_camera->view;
		culling_data->P00              = main_camera->projection[0][0];
		culling_data->P11              = main_camera->projection[1][1];
		culling_data->znear            = main_camera->near_plane;
		culling_data->zfar             = main_camera->far_plane;
		culling_data->meshlet_count    = Renderer::instance()->Meshlet_Count;
		culling_data->instance_count   = Renderer::instance()->Static_Instance_Count;
		culling_data->frustum_enable   = Renderer::instance()->Culling.frustum_culling;
		culling_data->backface_enable  = Renderer::instance()->Culling.backface_culling;
		culling_data->occlusion_enable = Renderer::instance()->Culling.occulsion_culling;
		culling_data->zbuffer_width    = static_cast<float>(Renderer::instance()->Last_Frame.hiz_buffer->getWidth());
		culling_data->zbuffer_height   = static_cast<float>(Renderer::instance()->Last_Frame.hiz_buffer->getHeight());
		Culling_Buffer.unmap();
	}

	// Check for update
	bool                  update         = false;
	std::atomic<uint32_t> instance_count = 0;

	auto group = Scene::instance()->getRegistry().group<cmpt::MeshletRenderer, cmpt::Tag, cmpt::Transform>();
	tbb::parallel_for_each(group.begin(), group.end(), [&instance_count, this](auto entity) {
		auto &mesh_renderer = Entity(entity).getComponent<cmpt::MeshletRenderer>();
		if (!Renderer::instance()->getResourceCache().hasModel(mesh_renderer.model))
		{
			return;
		}

		auto &model = Renderer::instance()->getResourceCache().loadModel(mesh_renderer.model);

		tbb::parallel_for_each(mesh_renderer.materials.begin(), mesh_renderer.materials.end(), [&instance_count](scope<IMaterial> &material) {
			instance_count.fetch_add(1, std::memory_order_relaxed);
		});
	});

	if (instance_count != Renderer::instance()->Static_Instance_Count || ResourceCache::update || cmpt::Renderable::update || cmpt::Tag::update)
	{
		cmpt::Renderable::update = false;
		cmpt::Tag::update        = false;

		Renderer::instance()->Indices_Count         = 0;
		Renderer::instance()->Static_Instance_Count = 0;
		Renderer::instance()->Meshlet_Count         = 0;

		std::vector<PerInstanceData> instance_data;
		std::vector<MaterialData>    material_data;
		std::vector<PerMeshletData>  meshlet_data;

		const auto group = Scene::instance()->getRegistry().group<>(entt::get<cmpt::MeshletRenderer, cmpt::Tag, cmpt::Transform>);
		group.each([&](const entt::entity &entity, const cmpt::MeshletRenderer &mesh_renderer, const cmpt::Tag &tag, const cmpt::Transform &transform) {
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

					meshlet_.instance_id = Renderer::instance()->Static_Instance_Count;

					meshlet_data.push_back(meshlet_);
				}

				PerInstanceData instance;

				// Instance Transform
				instance.world_transform = transform.world_transform;
				instance.pre_transform   = submesh.pre_transform;
				// Instance BoundingBox
				instance.bbox_min    = submesh.bounding_box.min_;
				instance.bbox_max    = submesh.bounding_box.max_;
				instance.material_id = std::numeric_limits<uint32_t>::max();
				instance.entity_id   = static_cast<uint32_t>(entity);

				MaterialData material_;
				if (material)
				{
					instance.material_id = static_cast<uint32_t>(material_data.size());

					material_.base_color          = material->base_color;
					material_.emissive_color      = material->emissive_color;
					material_.metallic_factor     = material->metallic_factor;
					material_.roughness_factor    = material->roughness_factor;
					material_.emissive_intensity  = material->emissive_intensity;
					material_.albedo_map          = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->albedo_map));
					material_.normal_map          = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->normal_map));
					material_.metallic_map        = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->metallic_map));
					material_.roughness_map       = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->roughness_map));
					material_.emissive_map        = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->emissive_map));
					material_.ao_map              = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->ao_map));
					material_.displacement_height = material->displacement_height;
					material_.displacement_map    = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material->displacement_map));
				}

				instance_data.push_back(instance);
				material_data.push_back(material_);

				Renderer::instance()->Static_Instance_Count++;
			}
		});

		// Enlarge buffer
		if (Instance_Buffer.getSize() < instance_data.size() * sizeof(PerInstanceData))
		{
			GraphicsContext::instance()->getQueueSystem().waitAll();
			Instance_Buffer            = Buffer(static_cast<VkDeviceSize>(static_cast<double>(instance_data.size() * sizeof(PerInstanceData)) * 1.1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			Instance_Visibility_Buffer = Buffer(static_cast<VkDeviceSize>(static_cast<double>(instance_data.size() * sizeof(uint32_t)) * 1.1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
			Material_Buffer            = Buffer(static_cast<VkDeviceSize>(static_cast<double>(material_data.size() * sizeof(MaterialData)) * 1.1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			update                     = true;
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
		std::memcpy(Material_Buffer.map(), material_data.data(), material_data.size() * sizeof(MaterialData));
		Instance_Buffer.unmap();
		Meshlet_Buffer.unmap();
		Material_Buffer.unmap();
	}
	else if (cmpt::Transform::update)
	{
		// Update transform only
		std::vector<PerInstanceData> instance_data(Renderer::instance()->Static_Instance_Count);
		std::memcpy(instance_data.data(), Instance_Buffer.map(), Renderer::instance()->Static_Instance_Count * sizeof(PerInstanceData));

		uint32_t   idx   = 0;
		const auto group = Scene::instance()->getRegistry().group<>(entt::get<cmpt::MeshletRenderer, cmpt::Tag, cmpt::Transform>);
		group.each([&](const entt::entity &entity, const cmpt::MeshletRenderer &mesh_renderer, const cmpt::Tag &tag, const cmpt::Transform &transform) {
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
		std::memcpy(Instance_Buffer.map(), instance_data.data(), Renderer::instance()->Static_Instance_Count * sizeof(PerInstanceData));

		Instance_Buffer.unmap();
	}

	return update;
}
}        // namespace Ilum