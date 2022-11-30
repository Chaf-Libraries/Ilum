#pragma once

#include "../Resource.hpp"

#include <Geometry/AABB.hpp>
#include <RHI/RHIContext.hpp>

#define MAX_BONE_INFLUENCE 4

namespace Ilum
{
template <>
class EXPORT_API Resource<ResourceType::Model> final : public IResource
{
  public:
	struct Meshlet
	{
		uint32_t meshlet_vertex_offset;
		uint32_t meshlet_primitive_offset;

		uint32_t vertex_count;
		uint32_t primitive_count;

		glm::vec3 center;
		glm::vec3 cone_axis;

		float radius;
		float cone_cutoff;
	};

	struct Vertex
	{
		alignas(16) glm::vec3 position;
		alignas(16) glm::vec3 normal;
		alignas(16) glm::vec3 tangent;
		alignas(16) glm::vec3 bitangent;

		glm::vec2 texcoord0;
		glm::vec2 texcoord1;

		int32_t bones[MAX_BONE_INFLUENCE]   = {-1};
		float   weights[MAX_BONE_INFLUENCE] = {0.f};
	};

	struct Bone
	{
		int32_t id;

		glm::mat4 offset;
	};

	struct Mesh
	{
		bool has_skeleton = false;

		// Vertex
		std::vector<Vertex> vertices;

		// Index
		std::vector<uint32_t> indices;

		// Meshlet
		std::vector<uint32_t> meshlet_vertices;
		std::vector<uint32_t> meshlet_primitives;
		std::vector<Meshlet>  meshlets;
	};

	struct Node
	{
		std::string name;

		int32_t mesh_id;
	};

  public:
	Resource(RHIContext *rhi_context, Mesh &&mesh);

	virtual ~Resource() override;

	const std::string &GetName() const;

	bool HasAnimation() const;

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