//#include "TriMesh.hpp"
//
//#include <algorithm>
//
//namespace Ilum
//{
//TriMesh::TriMesh(std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices) :
//    m_vertices(std::move(vertices)), m_indices(std::move(indices))
//{
//	std::vector<glm::vec3> positions(m_vertices.size());
//	std::transform(m_vertices.begin(), m_vertices.end(), positions.begin(), [](Vertex &v) -> glm::vec3 { return v.position; });
//	m_aabb.Merge(positions);
//}
//
//const std::vector<Vertex> &TriMesh::GetVertices() const
//{
//	return m_vertices;
//}
//
//const std::vector<uint32_t> &TriMesh::GetIndices() const
//{
//	return m_indices;
//}
//
//const AABB &TriMesh::GetAABB() const
//{
//	return m_aabb;
//}
//}        // namespace Ilum