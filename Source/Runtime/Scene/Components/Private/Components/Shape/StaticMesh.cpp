#pragma once

#include "Shape/StaticMesh.hpp"

#include <imgui.h>

namespace Ilum
{
namespace Cmpt
{
StaticMesh::StaticMesh(Node *node) :
    Shape("Static Mesh", node)
{
}

void StaticMesh::OnImGui()
{
	ImGui::Text("Vertices Count: %d", m_vertices_count);
	ImGui::Text("Triangle Count: %d", m_indices_count / 3);
	ImGui::Text("Meshlet Count: %d", m_meshlet_count);
}

std::type_index StaticMesh::GetType() const
{
	return typeid(StaticMesh);
}

void StaticMesh::SetMesh(RHIContext *rhi_context)
{
}

RHIBuffer *StaticMesh::GetVertexBuffer() const
{
	return m_vertex_buffer.get();
}

RHIBuffer *StaticMesh::GetIndexBuffer() const
{
	return m_index_buffer.get();
}

RHIBuffer *StaticMesh::GetMeshletBuffer() const
{
	return m_meshlet_buffer.get();
}

RHIBuffer *StaticMesh::GetInstanceBuffer() const
{
	return m_instance_buffer.get();
}

uint32_t StaticMesh::GetVerticesCount() const
{
	return m_vertices_count;
}

uint32_t StaticMesh::GetIndicesCount() const
{
	return m_indices_count;
}

uint32_t StaticMesh::GetMeshletCount() const
{
	return m_meshlet_count;
}
}        // namespace Cmpt
}        // namespace Ilum