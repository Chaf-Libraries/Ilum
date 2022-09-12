#pragma once

#include "Geometry/Vertex.hpp"

#include <vector>

namespace Ilum
{
class TriangleMesh
{
  public:
	TriangleMesh() = default;

	TriangleMesh(std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices);

	~TriangleMesh() = default;

	void SetVertices(std::vector<Vertex> &&vertices);

	void SetIndices(std::vector<uint32_t> &&indices);

	const std::vector<Vertex> &GetVertices() const;

	const std::vector<uint32_t> &GetIndices() const;

  private:
	std::vector<Vertex>   m_vertices;
	std::vector<uint32_t> m_indices;
};
}        // namespace Ilum