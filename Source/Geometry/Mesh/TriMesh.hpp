#pragma once

#include <vector>

#include "../Vertex.hpp"

namespace Ilum::Geo
{
class TriMesh
{
  public:
	TriMesh() = default;

	TriMesh(std::pmr::vector<Vertex> &&vertices, std::pmr::vector<uint32_t> &&indices);

	TriMesh(const TriMesh &) = delete;

	TriMesh &operator=(const TriMesh &) = delete;

	TriMesh(TriMesh &&other) noexcept;

	TriMesh &operator=(TriMesh &&other) noexcept;

	~TriMesh() = default;

	const std::pmr::vector<Vertex> &GetVertices() const;

	const std::pmr::vector<uint32_t> &GetIndices() const;

	std::pmr::vector<Vertex> &GetVertices();

	std::pmr::vector<uint32_t> &GetIndices();

	void Clear();

	bool Empty() const;

  private:
	std::pmr::vector<Vertex> m_vertices;
	std::pmr::vector<uint32_t> m_indices;
};
}        // namespace Ilum::Geo