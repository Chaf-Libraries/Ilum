#include "MaterialUpdate.hpp"

#include "Scene/Component/Renderable.hpp"
#include "Scene/Scene.hpp"

#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Profiler.hpp"

#include "File/FileSystem.hpp"

#include <tbb/tbb.h>

namespace Ilum::sym
{
void MaterialUpdate::run()
{
	GraphicsContext::instance()->getProfiler().beginSample("Material Update");

	if (Material::update)
	{
		auto meshlet_view = Scene::instance()->getRegistry().view<cmpt::StaticMeshRenderer>();

		// Collect instance data
		std::vector<size_t> material_offset(meshlet_view.size());
		size_t              material_count = 0;

		for (size_t i = 0; i < meshlet_view.size(); i++)
		{
			material_offset[i]     = material_count;
			auto &meshlet_renderer = Entity(meshlet_view[i]).getComponent<cmpt::StaticMeshRenderer>();
			material_count += meshlet_renderer.materials.size();
		}

		if (material_count * sizeof(MaterialData) > Renderer::instance()->Render_Buffer.Material_Buffer.getSize())
		{
			GraphicsContext::instance()->getQueueSystem().waitAll();
			Renderer::instance()->Render_Buffer.Material_Buffer = Buffer(material_count * sizeof(MaterialData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			Renderer::instance()->update();
		}

		MaterialData *material_data = reinterpret_cast<MaterialData *>(Renderer::instance()->Render_Buffer.Material_Buffer.map());

		// Update static mesh material
		tbb::parallel_for(tbb::blocked_range<size_t>(0, meshlet_view.size()), [&meshlet_view, &material_data, material_offset](const tbb::blocked_range<size_t> &r) {
			for (size_t i = r.begin(); i != r.end(); i++)
			{
				auto  entity           = Entity(meshlet_view[i]);
				auto &meshlet_renderer = entity.getComponent<cmpt::StaticMeshRenderer>();

				for (uint32_t material_id = 0; material_id < meshlet_renderer.materials.size(); material_id++)
				{
					auto &material = material_data[material_offset[i] + material_id];

					auto &material_ptr = meshlet_renderer.materials[material_id];

					material.base_color          = material_ptr.base_color;
					material.emissive_color      = material_ptr.emissive_color;
					material.metallic_factor     = material_ptr.metallic_factor;
					material.roughness_factor    = material_ptr.roughness_factor;
					material.emissive_intensity  = material_ptr.emissive_intensity;
					material.albedo_map          = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material_ptr.albedo_map));
					material.normal_map          = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material_ptr.normal_map));
					material.metallic_map        = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material_ptr.metallic_map));
					material.roughness_map       = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material_ptr.roughness_map));
					material.emissive_map        = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material_ptr.emissive_map));
					material.ao_map              = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material_ptr.ao_map));
					material.displacement_map    = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material_ptr.displacement_map));
					material.displacement_height = material_ptr.displacement_height;
				}
			}
		});
		Renderer::instance()->Render_Buffer.Material_Buffer.unmap();
	}
	Material::update = false;
	GraphicsContext::instance()->getProfiler().endSample("Material Update");
}
}        // namespace Ilum::sym