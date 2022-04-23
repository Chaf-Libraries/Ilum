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
		auto static_mesh_view  = Scene::instance()->getRegistry().view<cmpt::StaticMeshRenderer>();
		auto dynamic_mesh_view = Scene::instance()->getRegistry().view<cmpt::DynamicMeshRenderer>();

		// Collect instance data
		std::vector<size_t> material_offset(static_mesh_view.size());
		size_t              static_material_count = 0;

		for (size_t i = 0; i < static_mesh_view.size(); i++)
		{
			material_offset[i]         = static_material_count;
			auto &static_mesh_renderer = Entity(static_mesh_view[i]).getComponent<cmpt::StaticMeshRenderer>();
			static_material_count += static_mesh_renderer.materials.size();
		}

		if ((static_material_count + dynamic_mesh_view.size()) * sizeof(MaterialData) > Renderer::instance()->Render_Buffer.Material_Buffer.getSize())
		{
			GraphicsContext::instance()->getQueueSystem().waitAll();
			Renderer::instance()->Render_Buffer.Material_Buffer = Buffer((static_material_count + dynamic_mesh_view.size()) * sizeof(MaterialData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			Renderer::instance()->update();
		}

		MaterialData *material_data = reinterpret_cast<MaterialData *>(Renderer::instance()->Render_Buffer.Material_Buffer.map());

		// Update static mesh material
		//tbb::parallel_for(tbb::blocked_range<size_t>(0, static_mesh_view.size()), [&static_mesh_view, &material_data, material_offset](const tbb::blocked_range<size_t> &r) {
		for (size_t i = 0; i != static_mesh_view.size(); i++)
		//for (size_t i = r.begin(); i != r.end(); i++)
		{
			auto  entity               = Entity(static_mesh_view[i]);
			auto &static_mesh_renderer = entity.getComponent<cmpt::StaticMeshRenderer>();

			for (uint32_t material_id = 0; material_id < static_mesh_renderer.materials.size(); material_id++)
			{
				auto &material = material_data[material_offset[i] + material_id];

				auto &material_ptr = static_mesh_renderer.materials[material_id];

				material.base_color             = material_ptr.base_color;
				material.emissive_color         = material_ptr.emissive_color;
				material.emissive_intensity     = material_ptr.emissive_intensity;
				material.displacement           = material_ptr.displacement;
				material.subsurface             = material_ptr.subsurface;
				material.metallic               = material_ptr.metallic;
				material.specular               = material_ptr.specular;
				material.specular_tint          = material_ptr.specular_tint;
				material.roughness              = material_ptr.roughness;
				material.anisotropic            = material_ptr.anisotropic;
				material.sheen                  = material_ptr.sheen;
				material.sheen_tint             = material_ptr.sheen_tint;
				material.clearcoat              = material_ptr.clearcoat;
				material.clearcoat_gloss        = material_ptr.clearcoat_gloss;
				material.specular_transmission  = material_ptr.specular_transmission;
				material.diffuse_transmission   = material_ptr.diffuse_transmission;
				material.flatness               = material_ptr.flatness;
				material.thin                   = material_ptr.thin;
				material.refraction             = material_ptr.refraction;
				material.data                   = material_ptr.data;
				material.material_type          = static_cast<uint32_t>(material_ptr.type);

				for (uint32_t i = 0; i < static_cast<uint32_t>(TextureType::MaxNum); i++)
				{
					material.textures[i] = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material_ptr.textures[i]));
				}
			}
		}
		//});

		// Update dynamic mesh material
		tbb::parallel_for(tbb::blocked_range<size_t>(0, dynamic_mesh_view.size()), [&dynamic_mesh_view, &material_data, static_material_count](const tbb::blocked_range<size_t> &r) {
			for (size_t i = r.begin(); i != r.end(); i++)
			{
				auto  entity                = Entity(dynamic_mesh_view[i]);
				auto &dynamic_mesh_renderer = entity.getComponent<cmpt::DynamicMeshRenderer>();

				auto &material = material_data[static_material_count + i];

				auto &material_ptr = dynamic_mesh_renderer.material;

				material.base_color             = material_ptr.base_color;
				material.emissive_color         = material_ptr.emissive_color;
				material.emissive_intensity     = material_ptr.emissive_intensity;
				material.displacement           = material_ptr.displacement;
				material.subsurface             = material_ptr.subsurface;
				material.metallic               = material_ptr.metallic;
				material.specular               = material_ptr.specular;
				material.specular_tint          = material_ptr.specular_tint;
				material.roughness              = material_ptr.roughness;
				material.anisotropic            = material_ptr.anisotropic;
				material.sheen                  = material_ptr.sheen;
				material.sheen_tint             = material_ptr.sheen_tint;
				material.clearcoat              = material_ptr.clearcoat;
				material.clearcoat_gloss        = material_ptr.clearcoat_gloss;
				material.specular_transmission  = material_ptr.specular_transmission;
				material.diffuse_transmission   = material_ptr.diffuse_transmission;
				material.flatness               = material_ptr.flatness;
				material.thin                   = material_ptr.thin;
				material.refraction             = material_ptr.refraction;
				material.data                   = material_ptr.data;
				material.material_type          = static_cast<uint32_t>(material_ptr.type);
				for (uint32_t i = 0; i < static_cast<uint32_t>(TextureType::MaxNum); i++)
				{
					material.textures[i] = Renderer::instance()->getResourceCache().imageID(FileSystem::getRelativePath(material_ptr.textures[i]));
				}
			}
		});

		Renderer::instance()->Render_Buffer.Material_Buffer.unmap();
	}
	Material::update = false;
	GraphicsContext::instance()->getProfiler().endSample("Material Update");
}
}        // namespace Ilum::sym