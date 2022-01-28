#include "SubMesh.hpp"

namespace Ilum::Asset
{
uint32_t SubMesh::GetIndex() const
{
	return m_index;
}

glm::mat4 SubMesh::GetPreTransform() const
{
	return m_pre_transform;
}

uint32_t SubMesh::GetVerticesCount() const
{
	return m_vertices_count;
}

uint32_t SubMesh::GetIndicesCount() const
{
	return m_indices_count;
}

uint32_t SubMesh::GetMeshletCount() const
{
	return m_meshlet_count;
}

uint32_t SubMesh::GetVerticesOffset() const
{
	return m_vertices_offset;
}

uint32_t SubMesh::GetIndicesOffset() const
{
	return m_indices_offset;
}

uint32_t SubMesh::GetMeshletOffset() const
{
	return m_meshlet_offset;
}

const Geo::Bound &SubMesh::GetBound() const
{
	return m_bound;
}

const Material &SubMesh::GetMaterial() const
{
	return m_material;
}

Material &SubMesh::GetMaterial()
{
	return m_material;
}
}