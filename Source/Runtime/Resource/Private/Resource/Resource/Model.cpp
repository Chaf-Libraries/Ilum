#include "Resource/Model.hpp"

namespace Ilum
{
struct Resource<ResourceType::Model>::Impl
{
	std::string name;

	std::unique_ptr<RHIBuffer> vertex_buffer            = nullptr;
	std::unique_ptr<RHIBuffer> index_buffer             = nullptr;
	std::unique_ptr<RHIBuffer> meshlet_vertex_buffer    = nullptr;
	std::unique_ptr<RHIBuffer> meshlet_primitive_buffer = nullptr;
	std::unique_ptr<RHIBuffer> meshlet_buffer           = nullptr;

	std::vector<std::unique_ptr<RHIAccelerationStructure>> blas;
};

Resource<ResourceType::Model>::Resource(RHIContext *rhi_context, Mesh &&mesh)
{

}

Resource<ResourceType::Model>::~Resource()
{
	delete m_impl;
}

const std::string &Resource<ResourceType::Model>::GetName() const
{
	return m_impl->name;
}

bool Resource<ResourceType::Model>::HasAnimation() const
{
	return false;
}

RHIBuffer *Resource<ResourceType::Model>::GetVertexBuffer() const
{
	return m_impl->vertex_buffer.get();
}

RHIBuffer *Resource<ResourceType::Model>::GetIndexBuffer() const
{
	return m_impl->index_buffer.get();
}

RHIBuffer *Resource<ResourceType::Model>::GetMeshletVertexBuffer() const
{
	return m_impl->meshlet_vertex_buffer.get();
}

RHIBuffer *Resource<ResourceType::Model>::GetMeshletPrimitiveBuffer() const
{
	return m_impl->meshlet_primitive_buffer.get();
}

RHIBuffer *Resource<ResourceType::Model>::GetMeshletBuffer() const
{
	return m_impl->meshlet_buffer.get();
}

RHIAccelerationStructure *Resource<ResourceType::Model>::GetBLAS(uint32_t submesh_id) const
{
	return m_impl->blas.at(submesh_id).get();
}
}        // namespace Ilum