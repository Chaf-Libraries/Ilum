#pragma once

#include <glm/glm.hpp>

#include <Geometry/Bound.hpp>

#include "Material.hpp"

namespace Ilum::Asset
{
class SubMesh
{
	friend class Mesh;

  public:
	SubMesh() = default;

	~SubMesh() = default;

	uint32_t GetIndex() const;

	glm::mat4 GetPreTransform() const;

	uint32_t GetVerticesCount() const;

	uint32_t GetIndicesCount() const;

	uint32_t GetMeshletCount() const;

	uint32_t GetVerticesOffset() const;

	uint32_t GetIndicesOffset() const;

	uint32_t GetMeshletOffset() const;

	const Geo::Bound &GetBound() const;

	const Material &GetMaterial() const;

	Material &GetMaterial();

  private:
	uint32_t m_index = 0;

	glm::mat4 m_pre_transform = glm::mat4(1.f);

	uint32_t m_vertices_count = 0;
	uint32_t m_indices_count  = 0;
	uint32_t m_meshlet_count  = 0;

	uint32_t m_vertices_offset = 0;
	uint32_t m_indices_offset  = 0;
	uint32_t m_meshlet_offset  = 0;

	Material m_material;

	Geo::Bound m_bound;
};
}        // namespace Ilum::Asset