#pragma once

#include <vector>

#include "../Vertex.hpp"

namespace Ilum::Geo
{
class TriMesh
{
  public:
	TriMesh() = default;

	TriMesh(std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices);

	TriMesh(const TriMesh &) = delete;

	TriMesh &operator=(const TriMesh &) = delete;

	TriMesh(TriMesh &&other) noexcept;

	TriMesh &operator=(TriMesh &&other) noexcept;

	~TriMesh() = default;

	const std::vector<Vertex> &GetVertices() const;

	const std::vector<uint32_t> &GetIndices() const;

	void Clear();

	bool Empty() const;

  private:
	std::vector<Vertex>   m_vertices;
	std::vector<uint32_t> m_indices;
};
}        // namespace Ilum::Geo