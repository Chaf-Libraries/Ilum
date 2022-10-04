#pragma once

#include "Resource/Resource.hpp"

#include <Geometry/Bound/AABB.hpp>

namespace Ilum
{
class RHIBuffer;
class RHIAccelerationStructure;

STRUCT(Meshlet, Enable)
{
	STRUCT(Bound, Enable)
	{
		glm::vec3 center;
		float     radius;

		glm::vec3 cone_axis;
		float     cone_cutoff;
	}
	bound;

	uint32_t indices_offset;
	uint32_t indices_count;
	uint32_t vertices_offset;        // Global offset
	uint32_t vertices_count;

	alignas(16) uint32_t meshlet_vertices_offset;         // Meshlet offset
	uint32_t             meshlet_primitive_offset;        // Meshlet offset
};

STRUCT(Submesh, Enable)
{
	std::string name;

	uint32_t index;

	glm::mat4 pre_transform;

	AABB aabb;

	uint32_t vertices_count;
	uint32_t vertices_offset;
	uint32_t indices_count;
	uint32_t indices_offset;
	uint32_t meshlet_count;
	uint32_t meshlet_offset;
	// TODO: Material
};

template <>
class TResource<ResourceType::Model> : public Resource
{
  public:
	explicit TResource(size_t uuid);

	explicit TResource(size_t uuid, const std::string &meta, RHIContext *rhi_context);

	virtual ~TResource() override = default;

	virtual void Load(RHIContext *rhi_context) override;

	virtual void Import(RHIContext *rhi_context, const std::string &path) override;

	const std::string &GetName() const;

	const std::vector<Submesh> &GetSubmeshes() const;

	const AABB &GetAABB() const;

	RHIBuffer *GetVertexBuffer() const;

	RHIBuffer *GetIndexBuffer() const;

	RHIBuffer *GetMeshletVertexBuffer() const;

	RHIBuffer *GetMeshletPrimitiveBuffer() const;

	RHIBuffer *GetMeshletBuffer() const;

	RHIAccelerationStructure *GetBLAS(uint32_t submesh_id) const;

  private:
	std::string m_name;

	std::vector<Submesh> m_submeshes;

	AABB m_aabb;

	std::unique_ptr<RHIBuffer> m_vertex_buffer            = nullptr;
	std::unique_ptr<RHIBuffer> m_index_buffer             = nullptr;
	std::unique_ptr<RHIBuffer> m_meshlet_vertex_buffer    = nullptr;
	std::unique_ptr<RHIBuffer> m_meshlet_primitive_buffer = nullptr;
	std::unique_ptr<RHIBuffer> m_meshlet_buffer       = nullptr;

	std::vector<std::unique_ptr<RHIAccelerationStructure>> m_blas;
};
}        // namespace Ilum