#include "MeshletUpdate.hpp"

#include "Renderer/Renderer.hpp"

#include "Scene/Component/Renderable.hpp"

#include "Graphics/GraphicsContext.hpp"

#include <tbb/tbb.h>

namespace Ilum::sym
{
void MeshletUpdate::run()
{
	if (cmpt::Renderable::update)
	{
		auto meshlet_view = Scene::instance()->getRegistry().view<cmpt::MeshletRenderer>();
		auto mesh_view    = Scene::instance()->getRegistry().view<cmpt::MeshRenderer>();

		// Collect instance data
		std::vector<size_t> meshlet_offset(meshlet_view.size());
		std::vector<size_t> instance_offset(meshlet_view.size());

		Renderer::instance()->Render_Stats.static_instance_count = 0;
		Renderer::instance()->Render_Stats.meshlet_count         = 0;

		for (size_t i = 0; i < meshlet_view.size(); i++)
		{
			meshlet_offset[i]      = Renderer::instance()->Render_Stats.meshlet_count;
			instance_offset[i]     = Renderer::instance()->Render_Stats.static_instance_count;
			auto &meshlet_renderer = Entity(meshlet_view[i]).getComponent<cmpt::MeshletRenderer>();

			if (Renderer::instance()->getResourceCache().hasModel(meshlet_renderer.model))
			{
				auto model = Renderer::instance()->getResourceCache().loadModel(meshlet_renderer.model);
				Renderer::instance()->Render_Stats.meshlet_count += static_cast<uint32_t>(model.get().meshlets.size());
				Renderer::instance()->Render_Stats.static_instance_count += static_cast<uint32_t>(model.get().submeshes.size());
			}
		}

		if (Renderer::instance()->Render_Stats.meshlet_count * sizeof(MaterialData) > Renderer::instance()->Render_Buffer.Meshlet_Buffer.getSize())
		{
			GraphicsContext::instance()->getQueueSystem().waitAll();
			Renderer::instance()->Render_Buffer.Meshlet_Buffer = Buffer(Renderer::instance()->Render_Stats.meshlet_count * sizeof(PerMeshletData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			Renderer::instance()->Render_Buffer.Command_Buffer = Buffer(Renderer::instance()->Render_Stats.meshlet_count * sizeof(VkDrawIndexedIndirectCommand), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
			Renderer::instance()->Render_Buffer.Draw_Buffer    = Buffer(Renderer::instance()->Render_Stats.meshlet_count * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
			Renderer::instance()->update();
		}

		PerMeshletData *meshlet_data = reinterpret_cast<PerMeshletData *>(Renderer::instance()->Render_Buffer.Meshlet_Buffer.map());

		// Update static mesh material
		tbb::parallel_for(tbb::blocked_range<size_t>(0, meshlet_view.size()), [&meshlet_view, &meshlet_data, meshlet_offset, instance_offset](const tbb::blocked_range<size_t> &r) {
		for (size_t i = r.begin(); i != r.end(); i++)
		{
			auto        entity           = Entity(meshlet_view[i]);
			const auto &meshlet_renderer = entity.getComponent<cmpt::MeshletRenderer>();

			if (!Renderer::instance()->getResourceCache().hasModel(meshlet_renderer.model))
			{
				continue;
			}

			auto model = Renderer::instance()->getResourceCache().loadModel(meshlet_renderer.model);

			uint32_t meshlet_index = 0;
			for (uint32_t submesh_id = 0; submesh_id < model.get().submeshes.size(); submesh_id++)
			{
				const auto &submesh = model.get().submeshes[submesh_id];
				for (uint32_t meshlet_id = submesh.meshlet_offset; meshlet_id < submesh.meshlet_offset + submesh.meshlet_count; meshlet_id++)
				{
					auto &      meshlet     = meshlet_data[meshlet_offset[i] + meshlet_index++];
					const auto &meshlet_raw = model.get().meshlets[meshlet_id];
					meshlet.vertex_offset   = model.get().vertices_offset + meshlet_raw.vertices_offset;
					meshlet.index_offset    = model.get().indices_offset + meshlet_raw.indices_offset;
					meshlet.index_count     = meshlet_raw.indices_count;
					std::memcpy(&meshlet.center, meshlet_raw.bounds.center, 3 * sizeof(float));
					meshlet.radius = meshlet_raw.bounds.radius;
					std::memcpy(&meshlet.cone_apex, meshlet_raw.bounds.cone_apex, 3 * sizeof(float));
					meshlet.cone_cutoff = meshlet_raw.bounds.cone_cutoff;
					std::memcpy(&meshlet.cone_axis, meshlet_raw.bounds.cone_axis, 3 * sizeof(float));
					meshlet.instance_id = static_cast<uint32_t>(instance_offset[i] + submesh_id);
				}
			}
		}
		});
		Renderer::instance()->Render_Buffer.Meshlet_Buffer.unmap();
	}
	cmpt::Renderable::update = false;
}
}        // namespace Ilum::sym