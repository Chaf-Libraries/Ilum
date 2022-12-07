#include "Resource/Mesh.hpp"

namespace Ilum
{
struct Resource<ResourceType::Mesh>::Impl
{
	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;

	std::unique_ptr<RHIBuffer> vertex_buffer            = nullptr;
	std::unique_ptr<RHIBuffer> index_buffer             = nullptr;
	std::unique_ptr<RHIBuffer> meshlet_vertex_buffer    = nullptr;
	std::unique_ptr<RHIBuffer> meshlet_primitive_buffer = nullptr;
	std::unique_ptr<RHIBuffer> meshlet_buffer           = nullptr;

	std::unique_ptr<RHIAccelerationStructure> blas = nullptr;
};

Resource<ResourceType::Mesh>::Resource(RHIContext *rhi_context, const std::string &name, std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices) :
    IResource(name)
{
	m_impl = new Impl;
	Update(rhi_context, std::move(vertices), std::move(indices));
}

Resource<ResourceType::Mesh>::~Resource()
{
	delete m_impl;
}

RHIBuffer *Resource<ResourceType::Mesh>::GetVertexBuffer() const
{
	return m_impl->vertex_buffer.get();
}

RHIBuffer *Resource<ResourceType::Mesh>::GetIndexBuffer() const
{
	return m_impl->index_buffer.get();
}

RHIBuffer *Resource<ResourceType::Mesh>::GetMeshletVertexBuffer() const
{
	return m_impl->meshlet_vertex_buffer.get();
}

RHIBuffer *Resource<ResourceType::Mesh>::GetMeshletPrimitiveBuffer() const
{
	return m_impl->meshlet_primitive_buffer.get();
}

RHIBuffer *Resource<ResourceType::Mesh>::GetMeshletBuffer() const
{
	return m_impl->meshlet_buffer.get();
}

RHIAccelerationStructure *Resource<ResourceType::Mesh>::GetBLAS() const
{
	return m_impl->blas.get();
}

const std::vector<Resource<ResourceType::Mesh>::Vertex> &Resource<ResourceType::Mesh>::GetVertices() const
{
	return m_impl->vertices;
}

const std::vector<uint32_t> &Resource<ResourceType::Mesh>::GetIndices() const
{
	return m_impl->indices;
}

void Resource<ResourceType::Mesh>::Update(RHIContext *rhi_context, std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices)
{
	m_impl->vertices = std::move(vertices);
	m_impl->indices  = std::move(indices);
	UpdateBuffer(rhi_context);
}

void Resource<ResourceType::Mesh>::UpdateBuffer(RHIContext *rhi_context)
{
	auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Compute);
	cmd_buffer->Begin();

	m_impl->vertex_buffer = rhi_context->CreateBuffer<Vertex>(m_impl->vertices.size(), RHIBufferUsage::Vertex | RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);
	m_impl->index_buffer  = rhi_context->CreateBuffer<uint32_t>(m_impl->indices.size(), RHIBufferUsage::Index | RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);
	// m_impl->meshlet_vertex_buffer    = rhi_context->CreateBuffer<uint32_t>(m_impl->meshlet_vertices.size(), RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer,  RHIMemoryUsage::GPU_Only);
	// m_impl->meshlet_primitive_buffer = rhi_context->CreateBuffer<uint32_t>(m_impl->meshlet_primitives.size(), RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);
	// m_impl->meshlet_buffer           = rhi_context->CreateBuffer<Meshlet>(m_impl->meshlets.size(), RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);

	m_impl->vertex_buffer->CopyToDevice(m_impl->vertices.data(), m_impl->vertices.size() * sizeof(Vertex));
	m_impl->index_buffer->CopyToDevice(m_impl->indices.data(), m_impl->indices.size() * sizeof(uint32_t));
	// m_impl->meshlet_vertex_buffer->CopyToDevice(m_impl->meshlet_vertices.data(), m_impl->meshlet_vertices.size() * sizeof(uint32_t));
	// m_impl->meshlet_primitive_buffer->CopyToDevice(m_impl->meshlet_primitives.data(), m_impl->meshlet_primitives.size() * sizeof(uint32_t));
	// m_impl->meshlet_buffer->CopyToDevice(m_impl->meshlets.data(), m_impl->meshlets.size() * sizeof(Meshlet));

	m_impl->blas = rhi_context->CreateAcccelerationStructure();

	BLASDesc desc        = {};
	desc.name            = m_name;
	desc.vertex_buffer   = m_impl->vertex_buffer.get();
	desc.index_buffer    = m_impl->index_buffer.get();
	desc.vertices_count  = static_cast<uint32_t>(m_impl->vertices.size());
	desc.vertices_offset = 0;
	desc.indices_count   = static_cast<uint32_t>(m_impl->indices.size());
	desc.indices_offset  = 0;

	m_impl->blas->Update(cmd_buffer, desc);

	cmd_buffer->End();
	rhi_context->Execute(cmd_buffer);
}
}        // namespace Ilum