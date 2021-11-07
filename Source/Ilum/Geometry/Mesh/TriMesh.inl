#include "TriMesh.hpp"

namespace Ilum::geometry
{
template <typename Vertex>
TriMesh<Vertex>::TriMesh(std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices) :
    vertices(std::move(vertices)), indices(std::move(indices))
{
}

template <typename Vertex>
TriMesh<Vertex>::TriMesh(TriMesh &&other) noexcept :
    vertices(std::move(other.vertices)), indices(std::move(other.indices))
{
	other.clear();
}

template <typename Vertex>
TriMesh<Vertex> &TriMesh<Vertex>::operator=(TriMesh &&other) noexcept
{
	vertices = std::move(other.vertices);
	indices  = std::move(other.indices);
	other.clear();
	return *this;
}

template <typename Vertex>
void TriMesh<Vertex>::clear()
{
	vertices.clear();
	indices.clear();
}

template <typename Vertex>
bool TriMesh<Vertex>::empty() const
{
	return vertices.empty() && indices.empty();
}

template <typename Vertex>
void TriMesh<Vertex>::set(std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices)
{
	vertices = std::move(vertices);
	indices  = std::move(indices);
}
}        // namespace Ilum