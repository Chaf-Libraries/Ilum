#include "Resource/Model.hpp"

namespace Ilum
{
struct Resource<ResourceType::Model>::Impl
{
	std::string name;

	std::vector<Submesh> submeshes;

	std::unique_ptr<RHIBuffer> vertex_buffer            = nullptr;
	std::unique_ptr<RHIBuffer> index_buffer             = nullptr;
	std::unique_ptr<RHIBuffer> meshlet_vertex_buffer    = nullptr;
	std::unique_ptr<RHIBuffer> meshlet_primitive_buffer = nullptr;
	std::unique_ptr<RHIBuffer> meshlet_buffer           = nullptr;

	std::vector<std::unique_ptr<RHIAccelerationStructure>> blas;
};

Resource<ResourceType::Model>::Resource(
    const std::string &     name,
    RHIContext *            rhi_context,
    std::vector<Submesh> && submeshes,
    std::vector<Meshlet> && meshlets,
    std::vector<Vertex> &&  vertices,
    std::vector<uint32_t> &&indices,
    std::vector<uint32_t> &&meshlet_vertices,
    std::vector<uint32_t> &&meshlet_primitives)
{
	m_impl = new Impl;

	m_impl->name = name;

	m_impl->vertex_buffer            = rhi_context->CreateBuffer<Vertex>(vertices.size(), RHIBufferUsage::Transfer | RHIBufferUsage::Vertex | RHIBufferUsage::ShaderResource | RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::GPU_Only);
	m_impl->index_buffer             = rhi_context->CreateBuffer<uint32_t>(indices.size(), RHIBufferUsage::Transfer | RHIBufferUsage::Index | RHIBufferUsage::ShaderResource | RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::GPU_Only);
	m_impl->meshlet_vertex_buffer    = rhi_context->CreateBuffer<uint32_t>(meshlet_vertices.size(), RHIBufferUsage::Transfer | RHIBufferUsage::ShaderResource | RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::GPU_Only);
	m_impl->meshlet_primitive_buffer = rhi_context->CreateBuffer<uint32_t>(meshlet_primitives.size(), RHIBufferUsage::Transfer | RHIBufferUsage::ShaderResource | RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::GPU_Only);
	m_impl->meshlet_buffer           = rhi_context->CreateBuffer<Meshlet>(meshlets.size(), RHIBufferUsage::Transfer | RHIBufferUsage::ShaderResource | RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::GPU_Only);

	m_impl->vertex_buffer->CopyToDevice(vertices.data(), vertices.size() * sizeof(Vertex), 0);
	m_impl->index_buffer->CopyToDevice(indices.data(), indices.size() * sizeof(uint32_t), 0);
	m_impl->meshlet_vertex_buffer->CopyToDevice(meshlet_vertices.data(), meshlet_vertices.size() * sizeof(uint32_t), 0);
	m_impl->meshlet_primitive_buffer->CopyToDevice(meshlet_primitives.data(), meshlet_primitives.size() * sizeof(uint32_t), 0);
	m_impl->meshlet_buffer->CopyToDevice(meshlets.data(), meshlets.size() * sizeof(Meshlet), 0);

	// Build BLAS
	m_impl->blas.reserve(m_impl->submeshes.size());
	auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Compute);
	cmd_buffer->Begin();
	for (auto &submesh : m_impl->submeshes)
	{
		auto as = rhi_context->CreateAcccelerationStructure();

		BLASDesc desc        = {};
		desc.name            = m_impl->name;
		desc.vertex_buffer   = m_impl->vertex_buffer.get();
		desc.index_buffer    = m_impl->index_buffer.get();
		desc.vertices_count  = submesh.vertex_count;
		desc.vertices_offset = submesh.vertex_offset;
		desc.indices_count   = submesh.index_count;
		desc.indices_offset  = submesh.index_offset;

		as->Update(cmd_buffer, desc);
		m_impl->blas.emplace_back(std::move(as));
	}

	cmd_buffer->End();
	rhi_context->Submit({cmd_buffer});
}

Resource<ResourceType::Model>::~Resource()
{
	delete m_impl;
}

const std::string &Resource<ResourceType::Model>::GetName() const
{
	return m_impl->name;
}

const std::vector<Resource<ResourceType::Model>::Submesh>& Resource<ResourceType::Model>::GetSubmeshes() const
{
	return m_impl->submeshes;
}

RHIBuffer *Resource<ResourceType::Model>::GetVertexBuffer() const
{
	return m_impl->vertex_buffer.get();
}

RHIBuffer* Resource<ResourceType::Model>::GetIndexBuffer() const
{
	return m_impl->index_buffer.get();
}

RHIBuffer* Resource<ResourceType::Model>::GetMeshletVertexBuffer() const
{
	return m_impl->meshlet_vertex_buffer.get();
}

RHIBuffer* Resource<ResourceType::Model>::GetMeshletPrimitiveBuffer() const
{
	return m_impl->meshlet_primitive_buffer.get();
}

RHIBuffer* Resource<ResourceType::Model>::GetMeshletBuffer() const
{
	return m_impl->meshlet_buffer.get();
}

RHIAccelerationStructure* Resource<ResourceType::Model>::GetBLAS(uint32_t submesh_id) const
{
	return m_impl->blas.at(submesh_id).get();
}
}        // namespace Ilum