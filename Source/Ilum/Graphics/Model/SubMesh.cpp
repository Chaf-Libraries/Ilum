#include "SubMesh.hpp"

namespace Ilum
{
SubMesh::SubMesh(std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices, uint32_t index_offset) :
    m_vertices(std::move(vertices)),
    m_indices(std::move(indices)),
    m_index_offset(index_offset)
{
	for (auto& vertex : m_vertices)
	{
		m_bounding_box.merge(vertex.position);
	}
}

uint32_t SubMesh::getVertexCount() const
{
	return static_cast<uint32_t>(m_vertices.size());
}

uint32_t SubMesh::getIndexCount() const
{
	return static_cast<uint32_t>(m_indices.size());
}

uint32_t SubMesh::getIndexOffset() const
{
	return m_index_offset;
}

const std::vector<Vertex> &SubMesh::getVertices() const
{
	return m_vertices;
}

SubMesh::SubMesh(SubMesh &&other) noexcept :
    m_vertices(std::move(other.m_vertices)),
    m_indices(std::move(other.m_indices)),
    m_index_offset(other.m_index_offset),
    m_bounding_box(other.m_bounding_box)
{
	other.m_vertices.clear();
	other.m_indices.clear();
	other.m_index_offset   = 0;
}

SubMesh &SubMesh::operator=(SubMesh &&other) noexcept
{
	m_vertices       = std::move(other.m_vertices);
	m_indices        = std::move(other.m_indices);
	m_index_offset   = other.m_index_offset;
	m_bounding_box           = other.m_bounding_box;

	other.m_vertices.clear();
	other.m_indices.clear();
	other.m_index_offset   = 0;

	return *this;
}

const std::vector<uint32_t> &SubMesh::getIndices() const
{
	return m_indices;
}

const geometry::BoundingBox &SubMesh::getBoundingBox() const
{
	return m_bounding_box;
}

bool SubMesh::isVisible() const
{
	return m_visible;
}
}        // namespace Ilum