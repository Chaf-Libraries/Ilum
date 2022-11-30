#pragma once

#include "../Resource.hpp"

#include <Geometry/AABB.hpp>
#include <RHI/RHIContext.hpp>

namespace Ilum
{
template <>
class EXPORT_API Resource<ResourceType::Model> final : public IResource
{
  public:
	struct Vertex
	{
		alignas(16) glm::vec3 position;
		alignas(16) glm::vec3 normal;
		alignas(16) glm::vec3 tangent;
		alignas(16) glm::vec3 bitangent;

		glm::vec2 uv0;
		glm::vec2 uv1;

		int32_t bones[4]   = {-1};
		float   weights[4] = {0.f};
	};

	struct Mesh
	{
		std::vector<Vertex>   vertices;
		std::vector<uint32_t> indices;
	};

	struct Bone
	{
		int32_t id;
		glm::mat4 offset;
	};

	struct Submesh
	{
		std::string name;

		glm::mat4 pre_transform;

		uint32_t vertex_offset;
		uint32_t vertex_count;
		uint32_t index_offset;
		uint32_t index_count;
		uint32_t meshlet_offset;
		uint32_t meshlet_count;

		AABB aabb;
	};

	struct Meshlet
	{
		uint32_t vertex_offset;
		uint32_t vertex_count;
		uint32_t index_offset;
		uint32_t index_count;
		uint32_t meshlet_vertex_offset;
		uint32_t meshlet_primitive_offset;

		glm::vec3 center;
		float     radius;

		glm::vec3 cone_axis;
		float     cone_cutoff;
	};

  public:
	Resource(
	    const std::string &     name,
	    RHIContext *            rhi_context,
	    std::vector<Submesh> && submeshes,
	    std::vector<Meshlet> && meshlets,
	    std::vector<Vertex> &&  vertices,
	    std::vector<uint32_t> &&indices,
	    std::vector<uint32_t> &&meshlet_vertices,
	    std::vector<uint32_t> &&meshlet_primitives);

	virtual ~Resource() override;

	const std::string &GetName() const;

	bool HasAnimation() const;

	const std::vector<Submesh> &GetSubmeshes() const;

	RHIBuffer *GetVertexBuffer() const;

	RHIBuffer *GetIndexBuffer() const;

	RHIBuffer *GetMeshletVertexBuffer() const;

	RHIBuffer *GetMeshletPrimitiveBuffer() const;

	RHIBuffer *GetMeshletBuffer() const;

	RHIAccelerationStructure *GetBLAS(uint32_t submesh_id) const;

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum