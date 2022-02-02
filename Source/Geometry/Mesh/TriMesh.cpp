#include "TriMesh.hpp"

namespace Ilum::Geo
{
TriMesh::TriMesh(std::pmr::vector<Vertex> &&vertices, std::pmr::vector<uint32_t> &&indices) :
    m_vertices(std::move(vertices)), m_indices(std::move(indices))
{
}

TriMesh::TriMesh(TriMesh &&other) noexcept :
    m_vertices(std::move(other.m_vertices)), m_indices(std::move(other.m_indices))
{
	other.Clear();
}

TriMesh &TriMesh::operator=(TriMesh &&other) noexcept
{
	m_vertices = std::move(other.m_vertices);
	m_indices  = std::move(other.m_indices);
	other.Clear();
	return *this;
}

const std::pmr::vector<Vertex> &TriMesh::GetVertices() const
{
	return m_vertices;
}

const std::pmr::vector<uint32_t> &TriMesh::GetIndices() const
{
	return m_indices;
}

std::pmr::vector<Vertex> &TriMesh::GetVertices()
{
	return m_vertices;
}

std::pmr::vector<uint32_t> &TriMesh::GetIndices()
{
	return m_indices;
}

void TriMesh::Clear()
{
	m_vertices.clear();
	m_indices.clear();
}

bool TriMesh::Empty() const
{
	return m_vertices.empty() && m_indices.empty();
}
}        // namespace Ilum::Geo