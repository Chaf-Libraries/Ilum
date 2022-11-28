#pragma once

#include "AABB.hpp"
#include "Meshlet.hpp"
#include "Vertex.hpp"

#include <vector>

namespace Ilum
{
class TriMesh
{
  public:
	TriMesh(std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices);

	~TriMesh() = default;

	const std::vector<Vertex> &GetVertices() const;

	const std::vector<uint32_t> &GetIndices() const;

	const AABB &GetAABB() const;

  private:
	std::vector<Vertex> m_vertices;

	std::vector<uint32_t> m_indices;

	AABB m_aabb;
};
}        // namespace Ilum