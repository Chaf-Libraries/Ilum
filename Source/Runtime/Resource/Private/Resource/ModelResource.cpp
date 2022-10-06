#include "Resource/ModelResource.hpp"
#include "Importer/Model/ModelImporter.hpp"

#include <Core/Path.hpp>
#include <Core/Serialization.hpp>
#include <RHI/RHIContext.hpp>

namespace Ilum
{
TResource<ResourceType::Model>::TResource(size_t uuid) :
    Resource(uuid)
{
}

TResource<ResourceType::Model>::TResource(size_t uuid, const std::string &meta, RHIContext *rhi_context) :
    Resource(uuid, meta, rhi_context)
{
}

void TResource<ResourceType::Model>::Load(RHIContext *rhi_context, size_t index)
{
	std::string           name;
	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;
	std::vector<Meshlet>  meshlets;
	std::vector<uint32_t> meshlet_vertices;
	std::vector<uint32_t> meshlet_primitives;

	DESERIALIZE("Asset/Meta/" + std::to_string(m_uuid) + ".meta", ResourceType::Model, m_uuid, m_meta,
	          name, m_submeshes, vertices, indices, meshlets, meshlet_vertices, meshlet_primitives, m_aabb);

	m_vertex_buffer            = rhi_context->CreateBuffer<Vertex>(vertices.size(), RHIBufferUsage::Transfer | RHIBufferUsage::Vertex | RHIBufferUsage::ShaderResource | RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::GPU_Only);
	m_index_buffer             = rhi_context->CreateBuffer<uint32_t>(indices.size(), RHIBufferUsage::Transfer | RHIBufferUsage::Index | RHIBufferUsage::ShaderResource | RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::GPU_Only);
	m_meshlet_vertex_buffer    = rhi_context->CreateBuffer<uint32_t>(meshlet_vertices.size(), RHIBufferUsage::Transfer | RHIBufferUsage::ShaderResource | RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::GPU_Only);
	m_meshlet_primitive_buffer = rhi_context->CreateBuffer<uint32_t>(meshlet_primitives.size(), RHIBufferUsage::Transfer | RHIBufferUsage::ShaderResource | RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::GPU_Only);
	m_meshlet_buffer           = rhi_context->CreateBuffer<Meshlet>(meshlets.size(), RHIBufferUsage::Transfer | RHIBufferUsage::ShaderResource | RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::GPU_Only);

	m_vertex_buffer->CopyToDevice(vertices.data(), vertices.size() * sizeof(Vertex), 0);
	m_index_buffer->CopyToDevice(indices.data(), indices.size() * sizeof(uint32_t), 0);
	m_meshlet_vertex_buffer->CopyToDevice(meshlet_vertices.data(), meshlet_vertices.size() * sizeof(uint32_t), 0);
	m_meshlet_primitive_buffer->CopyToDevice(meshlet_primitives.data(), meshlet_primitives.size() * sizeof(uint32_t), 0);
	m_meshlet_buffer->CopyToDevice(meshlets.data(), meshlets.size() * sizeof(Meshlet), 0);

	// Build BLAS
	m_blas.reserve(m_submeshes.size());
	auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Compute);
	cmd_buffer->Begin();
	for (auto &submesh : m_submeshes)
	{
		auto as = rhi_context->CreateAcccelerationStructure();

		BLASDesc desc = {};

		desc.name            = m_name;
		desc.vertex_buffer   = m_vertex_buffer.get();
		desc.index_buffer    = m_index_buffer.get();
		desc.vertices_count  = submesh.vertices_count;
		desc.vertices_offset = submesh.vertices_offset;
		desc.indices_count   = submesh.indices_count;
		desc.indices_offset  = submesh.indices_offset;

		as->Update(cmd_buffer, desc);

		m_blas.emplace_back(std::move(as));
	}

	cmd_buffer->End();
	rhi_context->Submit({cmd_buffer});

	m_valid = true;
	m_index = index;
}

void TResource<ResourceType::Model>::Import(RHIContext *rhi_context, const std::string &path)
{
	size_t uuid = Hash(path);

	ModelImportInfo info = ModelImporter::Import(path);

	m_meta = fmt::format("Name: {}\nSubmeshes: {}\nVertices: {}\nTriangles: {}\nMeshlets: {}",
	                     Path::GetInstance().GetFileName(path), path, info.submeshes.size(), info.vertices.size(), info.indices.size() / 3, info.meshlets.size());

	SERIALIZE("Asset/Meta/" + std::to_string(uuid) + ".meta", ResourceType::Model, uuid, m_meta,
	          info.name, info.submeshes, info.vertices, info.indices, info.meshlets, info.meshlet_vertices, info.meshlet_primitives, info.aabb);

	for (auto &[tex_uuid, texture_info] : info.textures)
	{
		std::string meta = fmt::format("Name: {}\nOriginal Path: {}\nWidth: {}\nHeight: {}\nMips: {}\nLayers: {}\nFormat: {}",
		                               Path::GetInstance().GetFileName(path), path, texture_info.desc.width, texture_info.desc.height, texture_info.desc.mips, texture_info.desc.layers, rttr::type::get_by_name("RHIFormat").get_enumeration().value_to_name(texture_info.desc.format).to_string());
		SERIALIZE("Asset/Meta/" + std::to_string(tex_uuid) + ".meta", ResourceType::Texture, uuid, meta, texture_info.desc, texture_info.data);
	}
}

const std::string &TResource<ResourceType::Model>::GetName() const
{
	return m_name;
}

const std::vector<Submesh> &TResource<ResourceType::Model>::GetSubmeshes() const
{
	return m_submeshes;
}

const AABB &TResource<ResourceType::Model>::GetAABB() const
{
	return m_aabb;
}

RHIBuffer *TResource<ResourceType::Model>::GetVertexBuffer() const
{
	return m_vertex_buffer.get();
}

RHIBuffer *TResource<ResourceType::Model>::GetIndexBuffer() const
{
	return m_index_buffer.get();
}

RHIBuffer *TResource<ResourceType::Model>::GetMeshletVertexBuffer() const
{
	return m_meshlet_vertex_buffer.get();
}

RHIBuffer *TResource<ResourceType::Model>::GetMeshletPrimitiveBuffer() const
{
	return m_meshlet_primitive_buffer.get();
}

RHIBuffer *TResource<ResourceType::Model>::GetMeshletBuffer() const
{
	return m_meshlet_buffer.get();
}

RHIAccelerationStructure *TResource<ResourceType::Model>::GetBLAS(uint32_t submesh_id) const
{
	return m_submeshes.size() > submesh_id ? m_blas[submesh_id].get() : nullptr;
}
}        // namespace Ilum