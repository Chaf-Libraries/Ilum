//#include "GeometryUpdate.hpp"
//
//#include "Renderer/Renderer.hpp"
//
//#include "Graphics/Command/CommandBuffer.hpp"
//#include "Graphics/GraphicsContext.hpp"
//#include "Graphics/Profiler.hpp"
//
//#include "Scene/Component/Renderable.hpp"
//#include "Scene/Scene.hpp"
//
//#include <tbb/tbb.h>
//
//namespace Ilum::sym
//{
//void GeometryUpdate::run()
//{
//	GraphicsContext::instance()->getProfiler().beginSample("Geometry Update");
//	auto &resource_cache = Renderer::instance()->getResourceCache();
//
//	// Update static mesh only when it needed
//	auto &static_vertex_buffer = Renderer::instance()->Render_Buffer.Static_Vertex_Buffer;
//	auto &static_index_buffer  = Renderer::instance()->Render_Buffer.Static_Index_Buffer;
//	auto &meshlet_vertex_buffer = Renderer::instance()->Render_Buffer.Meshlet_Vertex_Buffer;
//	auto &meshlet_index_buffer = Renderer::instance()->Render_Buffer.Meshlet_Index_Buffer;
//
//	if (resource_cache.getVerticesCount() != 0 &&
//	    (resource_cache.getVerticesCount() * sizeof(Vertex) != static_vertex_buffer.getSize() ||
//	     resource_cache.getIndicesCount() * sizeof(uint32_t) != static_index_buffer.getSize()))
//	{
//		cmpt::StaticMeshRenderer::update = true;
//
//		GraphicsContext::instance()->getQueueSystem().waitAll();
//
//		// Resize buffer
//		static_vertex_buffer = Buffer(resource_cache.getVerticesCount() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_GPU_ONLY);
//		static_index_buffer  = Buffer(resource_cache.getIndicesCount() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_GPU_ONLY);
//		meshlet_vertex_buffer = Buffer(resource_cache.getMeshletVerticesCount() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
//		meshlet_index_buffer  = Buffer(resource_cache.getMeshletIndicesCount() * sizeof(uint8_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
//
//		if (static_vertex_buffer.getSize() == 0 || static_index_buffer.getSize() == 0)
//		{
//			return;
//		}
//
//		// Staging buffer
//		Buffer staging_vertex_buffer(static_vertex_buffer.getSize(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
//		Buffer staging_index_buffer(static_index_buffer.getSize(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
//		Buffer staging_meshlet_vertex_buffer(meshlet_vertex_buffer.getSize(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
//		Buffer staging_meshlet_index_buffer(meshlet_index_buffer.getSize(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
//
//		auto * vertex_data = staging_vertex_buffer.map();
//		auto  *index_data  = staging_index_buffer.map();
//		auto  *meshlet_vertex_data = staging_meshlet_vertex_buffer.map();
//		auto  *meshlet_index_data  = staging_meshlet_index_buffer.map();
//
//		// CPU -> Staging
//		for (auto &[name, index] : resource_cache.getModels())
//		{
//			auto &model = resource_cache.loadModel(name);
//
//			std::memcpy(vertex_data + model.get().vertices_offset * sizeof(Vertex), model.get().vertices.data(), sizeof(Ilum::Vertex) * model.get().vertices_count);
//			std::memcpy(index_data + model.get().indices_offset * sizeof(uint32_t), model.get().indices.data(), sizeof(uint32_t) * model.get().indices_count);
//			std::memcpy(meshlet_vertex_data + model.get().meshlet_vertices_offset * sizeof(uint32_t), model.get().meshlet_vertices.data(), sizeof(uint32_t) * model.get().meshlet_vertices_count);
//			std::memcpy(meshlet_index_data + model.get().meshlet_indices_offset * sizeof(uint8_t), model.get().meshlet_indices.data(), sizeof(uint8_t) * model.get().meshlet_indices_count);
//		}
//
//		staging_vertex_buffer.unmap();
//		staging_index_buffer.unmap();
//		staging_meshlet_vertex_buffer.unmap();
//		staging_meshlet_index_buffer.unmap();
//
//		// Staging -> GPU
//		CommandBuffer command_buffer(QueueUsage::Transfer);
//		command_buffer.begin();
//		command_buffer.copyBuffer(BufferInfo{staging_vertex_buffer}, BufferInfo{static_vertex_buffer}, static_vertex_buffer.getSize());
//		command_buffer.copyBuffer(BufferInfo{staging_index_buffer}, BufferInfo{static_index_buffer}, static_index_buffer.getSize());
//		command_buffer.copyBuffer(BufferInfo{staging_meshlet_vertex_buffer}, BufferInfo{meshlet_vertex_buffer}, meshlet_vertex_buffer.getSize());
//		command_buffer.copyBuffer(BufferInfo{staging_meshlet_index_buffer}, BufferInfo{meshlet_index_buffer}, meshlet_index_buffer.getSize());
//		command_buffer.end();
//		command_buffer.submitIdle();
//
//		// Update BLAS
//		for (auto &[name, index] : resource_cache.getModels())
//		{
//			auto &model = resource_cache.loadModel(name);
//
//			for (auto &submesh : model.get().submeshes)
//			{
//				VkAccelerationStructureGeometryKHR geometry_info             = {};
//				geometry_info.sType                                          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
//				geometry_info.flags                                          = VK_GEOMETRY_OPAQUE_BIT_KHR;
//				geometry_info.geometryType                                   = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
//				geometry_info.geometry.triangles.sType                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
//				geometry_info.geometry.triangles.vertexFormat                = VK_FORMAT_R32G32B32_SFLOAT;
//				geometry_info.geometry.triangles.vertexData.deviceAddress    = static_vertex_buffer.getDeviceAddress();
//				geometry_info.geometry.triangles.maxVertex                   = submesh.vertices_count;
//				geometry_info.geometry.triangles.vertexStride                = sizeof(Vertex);
//				geometry_info.geometry.triangles.indexType                   = VK_INDEX_TYPE_UINT32;
//				geometry_info.geometry.triangles.indexData.deviceAddress     = static_index_buffer.getDeviceAddress();
//				geometry_info.geometry.triangles.transformData.deviceAddress = 0;
//				geometry_info.geometry.triangles.transformData.hostAddress   = nullptr;
//
//				VkAccelerationStructureBuildRangeInfoKHR range_info = {};
//
//				range_info.primitiveCount  = submesh.indices_count / 3;
//				range_info.primitiveOffset = (model.get().indices_offset + submesh.indices_offset) * sizeof(uint32_t);
//				range_info.firstVertex     = model.get().vertices_offset + submesh.vertices_offset;
//				range_info.transformOffset = 0;
//
//				submesh.bottom_level_as.build(geometry_info, range_info);
//			}
//		}
//
//		Renderer::instance()->rebuild();
//
//		Material::update = true;
//	}
//
//	// Update dynamic mesh
//	auto mesh_view = Scene::instance()->getRegistry().view<cmpt::DynamicMeshRenderer>();
//
//	if (!mesh_view.empty())
//	{
//		Material::update = true;
//	}
//
//	tbb::parallel_for_each(mesh_view.begin(), mesh_view.end(), [](entt::entity entity) {
//		auto &mesh_renderer = Entity(entity).getComponent<cmpt::DynamicMeshRenderer>();
//		if (mesh_renderer.vertices.size() * sizeof(Vertex) != mesh_renderer.vertex_buffer.getSize())
//		{
//			GraphicsContext::instance()->getQueueSystem().waitAll();
//			mesh_renderer.vertex_buffer = Buffer(mesh_renderer.vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
//			mesh_renderer.need_update   = true;
//		}
//
//		if (mesh_renderer.indices.size() * sizeof(uint32_t) != mesh_renderer.index_buffer.getSize())
//		{
//			GraphicsContext::instance()->getQueueSystem().waitAll();
//			mesh_renderer.index_buffer = Buffer(mesh_renderer.indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
//			mesh_renderer.need_update  = true;
//		}
//
//		if (mesh_renderer.vertices.size() == 0 || mesh_renderer.indices.size() == 0)
//		{
//			mesh_renderer.need_update = false;
//			return;
//		}
//
//		if (mesh_renderer.need_update)
//		{
//			std::memcpy(mesh_renderer.vertex_buffer.map(), mesh_renderer.vertices.data(), mesh_renderer.vertex_buffer.getSize());
//			std::memcpy(mesh_renderer.index_buffer.map(), mesh_renderer.indices.data(), mesh_renderer.index_buffer.getSize());
//
//			mesh_renderer.vertex_buffer.unmap();
//			mesh_renderer.index_buffer.unmap();
//
//			mesh_renderer.need_update = false;
//		}
//	});
//	GraphicsContext::instance()->getProfiler().endSample("Geometry Update");
//}
//}        // namespace Ilum::sym