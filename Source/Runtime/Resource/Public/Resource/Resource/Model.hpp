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

	struct SkinnedVertex : public Vertex
	{
		int32_t bones[MAX_BONE_INFLUENCE]   = {-1};
		float   weights[MAX_BONE_INFLUENCE] = {0.f};
	};

	struct Bone
	{
		int32_t   id;
		glm::mat4 offset;
	};

	struct Mesh
	{
		std::string name;

		std::vector<Vertex>   vertices;
		std::vector<uint32_t> indices;
	};

	struct SkinnedMesh
	{
		std::string name;

		std::map<std::string, Bone> bones;
	};

	struct Node
	{
		glm::mat4 transform;
		uint32_t  mesh = ~0U;
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
	Resource(const std::string &name, RHIContext *rhi_context, std::vector<Mesh> &&meshes);

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