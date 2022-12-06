#pragma once

#include "../Resource.hpp"

#include <Geometry/AABB.hpp>
#include <RHI/RHIContext.hpp>

namespace Ilum
{
template <>
class EXPORT_API Resource<ResourceType::Mesh> final : public IResource
{
  public:
	struct Meshlet
	{
		uint32_t meshlet_vertex_offset;
		uint32_t meshlet_primitive_offset;

		uint32_t vertex_count;
		uint32_t primitive_count;

		glm::vec3 center;
		float     radius;

		glm::vec3 cone_axis;
		float     cone_cutoff;
	};

	struct Vertex
	{
		alignas(16) glm::vec3 position;
		alignas(16) glm::vec3 normal;
		alignas(16) glm::vec3 tangent;

		glm::vec2 texcoord0;
		glm::vec2 texcoord1;
	};

	struct Submesh
	{
		uint32_t vertex_offset = 0;
		uint32_t vertex_count  = 0;
		uint32_t index_offset  = 0;
		uint32_t index_count   = 0;
	};

	struct Mesh
	{
		std::vector<Vertex>   vertices;
		std::vector<uint32_t> indices;

		std::vector<Submesh> submeshes;
	};

	struct BoneInfo
	{
		uint32_t  id;
		glm::mat4 offset;
	};

	// struct Mesh
	//{
	//	std::string name;

	//	// Transform
	//	glm::mat4 transform;

	//	// Vertex
	//	std::vector<Vertex> vertices;

	//	// Index
	//	std::vector<uint32_t> indices;

	//	// Meshlet
	//	std::vector<uint32_t> meshlet_vertices;
	//	std::vector<uint32_t> meshlet_primitives;
	//	std::vector<Meshlet>  meshlets;

	//	// Skeleton animation
	//	std::map<std::string, Bone> bones;

	//	inline bool HasSkeleton() const
	//	{
	//		return !bones.empty();
	//	}
	//};

  public:
	Resource(RHIContext *rhi_context, const std::string &name, std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices);

	virtual ~Resource() override;

	const std::string &GetName() const;

	bool HasAnimation(uint32_t idx) const;

	const std::vector<Mesh> &GetMeshes() const;

	uint32_t GetMeshCount() const;

	RHIBuffer *GetVertexBuffer(uint32_t idx) const;

	RHIBuffer *GetIndexBuffer(uint32_t idx) const;

	RHIBuffer *GetMeshletVertexBuffer(uint32_t idx) const;

	RHIBuffer *GetMeshletPrimitiveBuffer(uint32_t idx) const;

	RHIBuffer *GetMeshletBuffer(uint32_t idx) const;

	RHIAccelerationStructure *GetBLAS(uint32_t idx) const;

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum