#include "Resource/ModelResource.hpp"

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

void TResource<ResourceType::Model>::Load(RHIContext *rhi_context)
{
}

void TResource<ResourceType::Model>::Import(RHIContext *rhi_context, const std::string &path)
{
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