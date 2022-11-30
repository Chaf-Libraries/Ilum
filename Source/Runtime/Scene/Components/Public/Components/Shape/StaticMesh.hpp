#pragma once

#include "Shape.hpp"

#include <Geometry/TriMesh.hpp>
#include <Material/Material.hpp>
#include <RHI/RHIContext.hpp>

namespace Ilum
{
namespace Cmpt
{
class EXPORT_API StaticMesh : public Shape
{
  public:
	struct Submesh
	{
		uint32_t vertex_offset;
		uint32_t index_offset;
		uint32_t vertex_count;
		uint32_t index_count;
		uint32_t meshlet_offset;
		uint32_t meshlet_count;
	};

  public:
	StaticMesh(Node *node);

	virtual ~StaticMesh() = default;

	virtual void OnImGui() override;

	virtual std::type_index GetType() const override;

	void SetMesh(RHIContext *rhi_context);

	RHIBuffer *GetVertexBuffer() const;

	RHIBuffer *GetIndexBuffer() const;

	RHIBuffer *GetMeshletBuffer() const;

	RHIBuffer *GetInstanceBuffer() const;

	uint32_t GetVerticesCount() const;

	uint32_t GetIndicesCount() const;

	uint32_t GetMeshletCount() const;

  private:
	std::unique_ptr<RHIBuffer> m_vertex_buffer  = nullptr;
	std::unique_ptr<RHIBuffer> m_index_buffer   = nullptr;
	std::unique_ptr<RHIBuffer> m_meshlet_buffer = nullptr;
	std::unique_ptr<RHIBuffer> m_instance_buffer = nullptr;

	uint32_t m_vertices_count = 0;
	uint32_t m_indices_count  = 0;
	uint32_t m_meshlet_count  = 0;
};
}        // namespace Cmpt
}        // namespace Ilum