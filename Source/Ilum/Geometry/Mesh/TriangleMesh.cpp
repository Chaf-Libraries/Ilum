#include "TriangleMesh.hpp"

namespace Ilum
{
TriangleMesh::TriangleMesh(std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices):
    m_vertices(std::move(vertices)), m_indices(std::move(indices))
{
}

void TriangleMesh::SetVertices(std::vector<Vertex> &&vertices)
{
	m_vertices = std::move(vertices);
}

void TriangleMesh::SetIndices(std::vector<uint32_t> &&indices)
{
	m_indices = std::move(indices);
}

const std::vector<Vertex> &TriangleMesh::GetVertices() const
{
	return m_vertices;
}

const std::vector<uint32_t> &TriangleMesh::GetIndices() const
{
	return m_indices;
}
}        // namespace Ilum