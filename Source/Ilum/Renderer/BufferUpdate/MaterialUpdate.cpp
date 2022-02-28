#include "MaterialUpdate.hpp"

#include "Material/PBR.h"

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
				auto        entity           = Entity(meshlet_view[i]);
				auto &meshlet_renderer = entity.getComponent<cmpt::StaticMeshRenderer>();

				for (uint32_t material_id = 0; material_id < meshlet_renderer.materials.size(); material_id++)
				{
					auto &material = material_data[material_offset[i] + material_id];

					auto &material_ptr = meshlet_renderer.materials[material_id];

					if (material_ptr && material_ptr->type() == typeid(material::PBRMaterial))
					{
						material::PBRMaterial *pbr = static_cast<material::PBRMaterial *>(material_ptr.get());

						material.base_color          = pbr->base_color;
						material.emissive_color      = pbr->emissive_color;
						material.metallic_factor     = pbr->metallic_factor;
						material.roughness_factor    = pbr->roughness_factor;
						material.emissive_intensity  = pbr->emissive_intensity;
						material.albedo_map          = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(pbr->albedo_map));
						material.normal_map          = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(pbr->normal_map));
						material.metallic_map        = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(pbr->metallic_map));
						material.roughness_map       = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(pbr->roughness_map));
						material.emissive_map        = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(pbr->emissive_map));
						material.ao_map              = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(pbr->ao_map));
						material.displacement_map    = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(pbr->displacement_map));
						material.displacement_height = pbr->displacement_height;
					}
					else
					{
						material.base_color          = glm::vec4(1.f);
						material.emissive_color      = glm::vec4(0.f);
						material.metallic_factor     = 1.f;
						material.roughness_factor    = 1.f;
						material.emissive_intensity  = 0.f;
						material.albedo_map          = std::numeric_limits<uint32_t>::max();
						material.normal_map          = std::numeric_limits<uint32_t>::max();
						material.metallic_map        = std::numeric_limits<uint32_t>::max();
						material.roughness_map       = std::numeric_limits<uint32_t>::max();
						material.emissive_map        = std::numeric_limits<uint32_t>::max();
						material.ao_map              = std::numeric_limits<uint32_t>::max();
						material.displacement_map    = std::numeric_limits<uint32_t>::max();
						material.displacement_height = 0.f;
					}
				}
			}
		});
		Renderer::instance()->Render_Buffer.Material_Buffer.unmap();
	}
	Material::update = false;
	GraphicsContext::instance()->getProfiler().endSample("Material Update");
}
}        // namespace Ilum::sym