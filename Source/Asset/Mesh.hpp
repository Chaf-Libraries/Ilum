#pragma once

#include "SubMesh.hpp"

#include <Geometry/Mesh/TriMesh.hpp>
#include <Geometry/Vertex.hpp>

#include <meshoptimizer.h>

#include <string>
#include <memory_resource>

namespace Ilum::Asset
{
struct Meshlet
{
	meshopt_Bounds bounds;
	uint32_t       indices_count;
	uint32_t       indices_offset;
	uint32_t       vertices_offset;
};

class Mesh
{
  public:
	Mesh() = default;

	Mesh(std::pmr::vector<Geo::Vertex> &&vertices, std::pmr::vector<uint32_t> &&indices);

	~Mesh() = default;

	Mesh(const Mesh &) = delete;

	Mesh &operator=(const Mesh &) = delete;

	Mesh(Mesh &&other) noexcept;

	Mesh &operator=(Mesh &&other) noexcept;

	static Mesh Create(const std::string &filename);

	static Mesh Create(std::pmr::vector<Geo::Vertex> &&vertices, std::pmr::vector<uint32_t> &&indices);

	const std::pmr::vector<SubMesh> &GetSubMesh() const;

	const std::pmr::vector<Meshlet> &GetMeshlet() const;

	const Geo::TriMesh &GetTriMesh() const;

	const Geo::Bound &GetBound() const;

	uint32_t GetVerticesCount() const;

	uint32_t GetIndicesCount() const;

	uint32_t GetVerticesOffset() const;

	uint32_t GetIndicesOffset() const;

	uint32_t GetHashValue() const;

	void SetVerticesOffset(uint32_t offset);

	void SetIndicesOffset(uint32_t offset);

	void Save(const std::string &filename);

  private:
	std::pmr::vector<SubMesh>  m_submeshes;
	std::pmr::vector<Meshlet> m_meshlets;

	Geo::TriMesh m_trimesh;
	Geo::Bound   m_bound;

	uint32_t m_vertices_count = 0;
	uint32_t m_indices_count  = 0;

	uint32_t m_vertices_offset = 0;
	uint32_t m_indices_offset  = 0;

	size_t m_hash = 0;
};
}        // namespace Ilum::Asset