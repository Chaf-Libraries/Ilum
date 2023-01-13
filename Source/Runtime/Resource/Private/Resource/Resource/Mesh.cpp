#include "Resource/Mesh.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
struct Resource<ResourceType::Mesh>::Impl
{
	size_t vertex_count  = 0;
	size_t index_count   = 0;
	size_t meshlet_count = 0;

	std::unique_ptr<RHIBuffer> vertex_buffer       = nullptr;
	std::unique_ptr<RHIBuffer> index_buffer        = nullptr;
	std::unique_ptr<RHIBuffer> meshlet_data_buffer = nullptr;
	std::unique_ptr<RHIBuffer> meshlet_buffer      = nullptr;

	std::unique_ptr<RHIAccelerationStructure> blas = nullptr;
};

Resource<ResourceType::Mesh>::Resource(RHIContext *rhi_context, const std::string &name) :
    IResource(rhi_context, name, ResourceType::Mesh)
{
}

Resource<ResourceType::Mesh>::Resource(RHIContext *rhi_context, const std::string &name, std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices, std::vector<Meshlet> &&meshlets, std::vector<uint32_t> &&meshlet_data) :
    IResource(name)
{
	m_impl = new Impl;
	Update(rhi_context, std::move(vertices), std::move(indices), std::move(meshlets), std::move(meshlet_data));
}

Resource<ResourceType::Mesh>::~Resource()
{
	delete m_impl;
}

bool Resource<ResourceType::Mesh>::Validate() const
{
	return m_impl != nullptr;
}

void Resource<ResourceType::Mesh>::Load(RHIContext *rhi_context)
{
	m_impl = new Impl;

	std::vector<uint8_t>  thumbnail_data;
	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;
	std::vector<Meshlet>  meshlets;
	std::vector<uint32_t> meshlet_data;

	DESERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Mesh), thumbnail_data, vertices, indices, meshlets, meshlet_data);

	Update(rhi_context, std::move(vertices), std::move(indices), std::move(meshlets), std::move(meshlet_data));
}

RHIBuffer *Resource<ResourceType::Mesh>::GetVertexBuffer() const
{
	return m_impl->vertex_buffer.get();
}

RHIBuffer *Resource<ResourceType::Mesh>::GetIndexBuffer() const
{
	return m_impl->index_buffer.get();
}

RHIBuffer *Resource<ResourceType::Mesh>::GetMeshletDataBuffer() const
{
	return m_impl->meshlet_data_buffer.get();
}

RHIBuffer *Resource<ResourceType::Mesh>::GetMeshletBuffer() const
{
	return m_impl->meshlet_buffer.get();
}

RHIAccelerationStructure *Resource<ResourceType::Mesh>::GetBLAS() const
{
	return m_impl->blas.get();
}

size_t Resource<ResourceType::Mesh>::GetVertexCount() const
{
	return m_impl->vertex_count;
}

size_t Resource<ResourceType::Mesh>::GetIndexCount() const
{
	return m_impl->index_count;
}

size_t Resource<ResourceType::Mesh>::GetMeshletCount() const
{
	return m_impl->meshlet_count;
}

void Resource<ResourceType::Mesh>::Update(RHIContext *rhi_context, std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices, std::vector<Meshlet> &&meshlets, std::vector<uint32_t> &&meshlet_data)
{
	m_impl->vertex_count  = vertices.size();
	m_impl->index_count   = indices.size();
	m_impl->meshlet_count = meshlets.size();

	m_impl->vertex_buffer       = rhi_context->CreateBuffer<Vertex>(vertices.size(), RHIBufferUsage::Vertex | RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);
	m_impl->index_buffer        = rhi_context->CreateBuffer<uint32_t>(indices.size(), RHIBufferUsage::Index | RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);
	m_impl->meshlet_data_buffer = rhi_context->CreateBuffer<uint32_t>(meshlet_data.size(), RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);
	m_impl->meshlet_buffer      = rhi_context->CreateBuffer<Meshlet>(meshlets.size(), RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);

	m_impl->vertex_buffer->CopyToDevice(vertices.data(), vertices.size() * sizeof(Vertex));
	m_impl->index_buffer->CopyToDevice(indices.data(), indices.size() * sizeof(uint32_t));
	m_impl->meshlet_data_buffer->CopyToDevice(meshlet_data.data(), meshlet_data.size() * sizeof(uint32_t));
	m_impl->meshlet_buffer->CopyToDevice(meshlets.data(), meshlets.size() * sizeof(Meshlet));

	m_impl->blas = rhi_context->CreateAcccelerationStructure();

	BLASDesc desc        = {};
	desc.name            = m_name;
	desc.vertex_buffer   = m_impl->vertex_buffer.get();
	desc.index_buffer    = m_impl->index_buffer.get();
	desc.vertices_count  = static_cast<uint32_t>(vertices.size());
	desc.vertices_offset = 0;
	desc.indices_count   = static_cast<uint32_t>(indices.size());
	desc.indices_offset  = 0;

	auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Compute);
	cmd_buffer->Begin();
	m_impl->blas->Update(cmd_buffer, desc);
	cmd_buffer->End();
	rhi_context->Execute(cmd_buffer);

	std::vector<uint8_t> thumbnail_data;
	SERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Mesh), thumbnail_data, vertices, indices, meshlets, meshlet_data);
}
}        // namespace Ilum