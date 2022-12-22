#pragma once

#include "../Resource.hpp"

#include <Geometry/Meshlet.hpp>

namespace Ilum
{
class RHIContext;
class RHIBuffer;
class RHIAccelerationStructure;

template <>
class EXPORT_API Resource<ResourceType::Mesh> final : public IResource
{
  public:
	struct Vertex
	{
		alignas(16) glm::vec3 position;
		alignas(16) glm::vec3 normal;
		alignas(16) glm::vec3 tangent;

		alignas(16) glm::vec2 texcoord0;
		glm::vec2 texcoord1;
	};

  public:
	Resource(RHIContext *rhi_context, const std::string &name, std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices, std::vector<Meshlet> &&meshlets, std::vector<uint32_t> &&meshlet_data);

	virtual ~Resource() override;

	RHIBuffer *GetVertexBuffer() const;

	RHIBuffer *GetIndexBuffer() const;

	RHIBuffer *GetMeshletDataBuffer() const;

	RHIBuffer *GetMeshletBuffer() const;

	RHIAccelerationStructure *GetBLAS() const;

	size_t GetVertexCount() const;

	size_t GetIndexCount() const;

	size_t GetMeshletCount() const;

	void Update(RHIContext *rhi_context, std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices, std::vector<Meshlet> &&meshlets, std::vector<uint32_t> &&meshlet_data);

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum