#pragma once

#include "Geometry/Bound/AABB.hpp"
#include "Geometry/Mesh/TriMesh.hpp"
#include "Vertex.hpp"

namespace Ilum
{
struct SubMesh
{
  public:
	SubMesh(std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices, uint32_t index_offset);

	~SubMesh() = default;

	SubMesh(const SubMesh &) = delete;

	SubMesh &operator=(const SubMesh &) = delete;

	SubMesh(SubMesh &&other) noexcept;

	SubMesh &operator=(SubMesh &&other) noexcept;

	uint32_t getVertexCount() const;

	uint32_t getIndexCount() const;

	uint32_t getIndexOffset() const;

	const std::vector<Vertex> &getVertices() const;

	const std::vector<uint32_t> &getIndices() const;

	const AABB &getAABB() const;

  private:
	uint32_t m_index_offset   = 0;

	std::vector<Vertex>   m_vertices;
	std::vector<uint32_t> m_indices;

	AABB m_aabb;
};
}        // namespace Ilum