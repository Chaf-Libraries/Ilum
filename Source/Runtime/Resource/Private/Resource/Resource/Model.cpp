#include "Resource/Model.hpp"

namespace Ilum
{
struct Resource<ResourceType::Model>::Impl
{
	std::string name;

	std::vector<Mesh> meshes;

	std::vector<std::unique_ptr<RHIBuffer>> vertex_buffer;
	std::vector<std::unique_ptr<RHIBuffer>> index_buffer;
	std::vector<std::unique_ptr<RHIBuffer>> meshlet_vertex_buffer;
	std::vector<std::unique_ptr<RHIBuffer>> meshlet_primitive_buffer;
	std::vector<std::unique_ptr<RHIBuffer>> meshlet_buffer;

	std::vector<std::unique_ptr<RHIAccelerationStructure>> blas;
};

Resource<ResourceType::Model>::Resource(const std::string &name, RHIContext *rhi_context, std::vector<Mesh> &&meshes)
{
	m_impl         = new Impl;
	m_impl->name   = name;
	m_impl->meshes = std::move(meshes);

	m_impl->vertex_buffer.reserve(m_impl->meshes.size());
	m_impl->index_buffer.reserve(m_impl->meshes.size());
	m_impl->meshlet_vertex_buffer.reserve(m_impl->meshes.size());
	m_impl->meshlet_primitive_buffer.reserve(m_impl->meshes.size());
	m_impl->meshlet_buffer.reserve(m_impl->meshes.size());
	m_impl->blas.reserve(m_impl->meshes.size());

	//auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Compute);
	//cmd_buffer->Begin();

	for (const auto &mesh : m_impl->meshes)
	{
		std::unique_ptr<RHIBuffer> vertex_buffer            = nullptr;
		std::unique_ptr<RHIBuffer> index_buffer             = nullptr;
		std::unique_ptr<RHIBuffer> meshlet_vertex_buffer    = nullptr;
		std::unique_ptr<RHIBuffer> meshlet_primitive_buffer = nullptr;
		std::unique_ptr<RHIBuffer> meshlet_buffer           = nullptr;

		std::unique_ptr<RHIAccelerationStructure> blas = nullptr;

		//vertex_buffer            = rhi_context->CreateBuffer<Vertex>(mesh.vertices.size(), RHIBufferUsage::Vertex | RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, mesh.HasSkeleton() ? RHIMemoryUsage::CPU_TO_GPU : RHIMemoryUsage::GPU_Only);
		//index_buffer             = rhi_context->CreateBuffer<uint32_t>(mesh.indices.size(), RHIBufferUsage::Index | RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, mesh.HasSkeleton() ? RHIMemoryUsage::CPU_TO_GPU : RHIMemoryUsage::GPU_Only);
		//meshlet_vertex_buffer    = rhi_context->CreateBuffer<uint32_t>(mesh.meshlet_vertices.size(), RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, mesh.HasSkeleton() ? RHIMemoryUsage::CPU_TO_GPU : RHIMemoryUsage::GPU_Only);
		//meshlet_primitive_buffer = rhi_context->CreateBuffer<uint32_t>(mesh.meshlet_primitives.size(), RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, mesh.HasSkeleton() ? RHIMemoryUsage::CPU_TO_GPU : RHIMemoryUsage::GPU_Only);
		//meshlet_buffer           = rhi_context->CreateBuffer<Meshlet>(mesh.meshlets.size(), RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, mesh.HasSkeleton() ? RHIMemoryUsage::CPU_TO_GPU : RHIMemoryUsage::GPU_Only);

		//vertex_buffer->CopyToDevice(mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex));
		//index_buffer->CopyToDevice(mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t));
		//meshlet_vertex_buffer->CopyToDevice(mesh.meshlet_vertices.data(), mesh.meshlet_vertices.size() * sizeof(uint32_t));
		//meshlet_primitive_buffer->CopyToDevice(mesh.meshlet_primitives.data(), mesh.meshlet_primitives.size() * sizeof(uint32_t));
		//meshlet_buffer->CopyToDevice(mesh.meshlets.data(), mesh.meshlets.size() * sizeof(Meshlet));

		// TODO: Ray Tracing for skeleton mesh
		blas = rhi_context->CreateAcccelerationStructure();

		BLASDesc desc        = {};
		desc.name            = mesh.name;
		desc.vertex_buffer   = vertex_buffer.get();
		desc.index_buffer    = index_buffer.get();
		desc.vertices_count  = static_cast<uint32_t>(mesh.vertices.size());
		desc.vertices_offset = 0;
		desc.indices_count   = static_cast<uint32_t>(mesh.indices.size());
		desc.indices_offset  = 0;

		//blas->Update(cmd_buffer, desc);

		m_impl->vertex_buffer.emplace_back(std::move(vertex_buffer));
		m_impl->index_buffer.emplace_back(std::move(index_buffer));
		m_impl->meshlet_vertex_buffer.emplace_back(std::move(meshlet_vertex_buffer));
		m_impl->meshlet_primitive_buffer.emplace_back(std::move(meshlet_primitive_buffer));
		m_impl->meshlet_buffer.emplace_back(std::move(meshlet_buffer));
		m_impl->blas.emplace_back(std::move(blas));
	}

	//cmd_buffer->End();
	//rhi_context->Execute(cmd_buffer);
}

Resource<ResourceType::Model>::~Resource()
{
	delete m_impl;
}

const std::string &Resource<ResourceType::Model>::GetName() const
{
	return m_impl->name;
}

bool Resource<ResourceType::Model>::HasAnimation(uint32_t idx) const
{
	//return !m_impl->meshes.at(idx).bones.empty();
	return false;
}

const std::vector<Resource<ResourceType::Model>::Mesh> &Resource<ResourceType::Model>::GetMeshes() const
{
	return m_impl->meshes;
}

uint32_t Resource<ResourceType::Model>::GetMeshCount() const
{
	return static_cast<uint32_t>(m_impl->meshes.size());
}

RHIBuffer *Resource<ResourceType::Model>::GetVertexBuffer(uint32_t idx) const
{
	return m_impl->vertex_buffer.at(idx).get();
}

RHIBuffer *Resource<ResourceType::Model>::GetIndexBuffer(uint32_t idx) const
{
	return m_impl->index_buffer.at(idx).get();
}

RHIBuffer *Resource<ResourceType::Model>::GetMeshletVertexBuffer(uint32_t idx) const
{
	return m_impl->meshlet_vertex_buffer.at(idx).get();
}

RHIBuffer *Resource<ResourceType::Model>::GetMeshletPrimitiveBuffer(uint32_t idx) const
{
	return m_impl->meshlet_primitive_buffer.at(idx).get();
}

RHIBuffer *Resource<ResourceType::Model>::GetMeshletBuffer(uint32_t idx) const
{
	return m_impl->meshlet_buffer.at(idx).get();
}

RHIAccelerationStructure *Resource<ResourceType::Model>::GetBLAS(uint32_t idx) const
{
	return m_impl->blas.at(idx).get();
}
}        // namespace Ilum