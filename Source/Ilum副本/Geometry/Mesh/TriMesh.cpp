#include "TriMesh.hpp"

namespace Ilum::geometry
{
TriMesh::TriMesh(std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices) :
    vertices(std::move(vertices)), indices(std::move(indices))
{
}

TriMesh::TriMesh(TriMesh &&other) noexcept :
    vertices(std::move(other.vertices)), indices(std::move(other.indices))
{
	other.clear();
}

TriMesh &TriMesh::operator=(TriMesh &&other) noexcept
{
	vertices = std::move(other.vertices);
	indices  = std::move(other.indices);
	other.clear();
	return *this;
}

void TriMesh::clear()
{
	vertices.clear();
	indices.clear();
}

bool TriMesh::empty() const
{
	return vertices.empty() && indices.empty();
}
}        // namespace Ilum