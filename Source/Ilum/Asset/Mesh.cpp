#pragma once

#include <RHI/Command.hpp>
#include <RHI/ImGuiContext.hpp>

#include "AssetManager.hpp"
#include "Material.hpp"
#include "Mesh.hpp"

#include <imgui.h>

#include <glm/gtc/type_ptr.hpp>

namespace Ilum
{
Submesh::Submesh(RHIDevice *device, AssetManager &manager) :
    p_device(device),
    m_manager(manager)
{
}

const std::string &Submesh::GetName() const
{
	return m_name;
}

bool Submesh::OnImGui(ImGuiContext &context)
{
	ImGui::BulletText("Vertex Count: %ld", m_vertices.size());
	ImGui::BulletText("Index Count: %ld", m_indices.size());
	ImGui::BulletText("Meshlet Count: %ld", m_meshlets.size());

	bool is_update = false;

	ImGui::PushID(this);
	if (ImGui::TreeNode("Material"))
	{
		bool hit_button = false;
		ImGui::PushItemWidth(ImGui::GetContentRegionAvailWidth() * 0.4f);
		if (m_material && m_manager.IsValid(m_material))
		{
			hit_button = ImGui::Button(m_material->GetName().c_str());
		}
		else
		{
			hit_button = ImGui::Button("", ImVec2(100, 0));
		}
		ImGui::PopItemWidth();
		if (ImGui::BeginDragDropTarget())
		{
			if (const auto *pay_load = ImGui::AcceptDragDropPayload("Material"))
			{
				ASSERT(pay_load->DataSize == sizeof(uint32_t));
				if (m_manager.GetIndex(m_material) != *static_cast<uint32_t *>(pay_load->Data))
				{
					m_material = m_manager.GetMaterial(*static_cast<uint32_t *>(pay_load->Data));
					is_update  = true;
				}
			}
			ImGui::EndDragDropTarget();
		}

		if (m_material)
		{
			if (m_material->OnImGui(context))
			{
				is_update = true;
			}
		}

		if (hit_button)
		{
			m_material = nullptr;
			is_update  = true;
		}

		ImGui::TreePop();
	}
	ImGui::PopID();

	return is_update;
}

void Submesh::UpdateBuffer()
{
	if (!m_bottom_level_acceleration_structure)
	{
		m_bottom_level_acceleration_structure = std::make_unique<AccelerationStructure>(p_device);
	}

	if (!m_vertex_buffer || m_vertex_buffer->GetSize() < m_vertices.size() * sizeof(ShaderInterop::Vertex))
	{
		m_vertex_buffer = std::make_unique<Buffer>(
		    p_device,
		    BufferDesc(
		        sizeof(ShaderInterop::Vertex),
		        static_cast<uint32_t>(m_vertices.size()),
		        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
		        VMA_MEMORY_USAGE_GPU_ONLY));
	}

	if (!m_index_buffer || m_index_buffer->GetSize() < m_indices.size() * sizeof(uint32_t))
	{
		m_index_buffer = std::make_unique<Buffer>(
		    p_device,
		    BufferDesc(
		        sizeof(uint32_t),
		        static_cast<uint32_t>(m_indices.size()),
		        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
		        VMA_MEMORY_USAGE_GPU_ONLY));
	}

	if (!m_meshlet_vertex_buffer || m_meshlet_vertex_buffer->GetSize() < m_meshlet_vertices.size() * sizeof(uint32_t))
	{
		m_meshlet_vertex_buffer = std::make_unique<Buffer>(
		    p_device,
		    BufferDesc(
		        sizeof(uint32_t),
		        static_cast<uint32_t>(m_meshlet_vertices.size()),
		        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		        VMA_MEMORY_USAGE_GPU_ONLY));
	}

	if (!m_meshlet_triangle_buffer || m_meshlet_triangle_buffer->GetSize() < m_meshlet_triangles.size() * sizeof(uint32_t))
	{
		m_meshlet_triangle_buffer = std::make_unique<Buffer>(
		    p_device,
		    BufferDesc(
		        sizeof(uint32_t),
		        static_cast<uint32_t>(m_meshlet_triangles.size()),
		        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		        VMA_MEMORY_USAGE_GPU_ONLY));
	}

	if (!m_meshlet_buffer || m_meshlet_buffer->GetSize() < m_meshlets.size() * sizeof(ShaderInterop::Meshlet))
	{
		m_meshlet_buffer = std::make_unique<Buffer>(
		    p_device,
		    BufferDesc(
		        sizeof(ShaderInterop::Meshlet),
		        static_cast<uint32_t>(m_meshlets.size()),
		        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		        VMA_MEMORY_USAGE_GPU_ONLY));
	}

	if (!m_meshlet_bound_buffer || m_meshlet_bound_buffer->GetSize() < m_meshlet_bounds.size() * sizeof(ShaderInterop::MeshletBound))
	{
		m_meshlet_bound_buffer = std::make_unique<Buffer>(
		    p_device,
		    BufferDesc(
		        sizeof(ShaderInterop::MeshletBound),
		        static_cast<uint32_t>(m_meshlet_bounds.size()),
		        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		        VMA_MEMORY_USAGE_GPU_ONLY));
	}

	Buffer vertex_buffer_staging(
	    p_device,
	    BufferDesc(
	        sizeof(ShaderInterop::Vertex),
	        static_cast<uint32_t>(m_vertices.size()),
	        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	        VMA_MEMORY_USAGE_CPU_TO_GPU));

	Buffer index_staging(
	    p_device,
	    BufferDesc(
	        sizeof(uint32_t),
	        static_cast<uint32_t>(m_indices.size()),
	        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	        VMA_MEMORY_USAGE_CPU_TO_GPU));

	Buffer meshlet_vertex_staging(
	    p_device,
	    BufferDesc(
	        sizeof(uint32_t),
	        static_cast<uint32_t>(m_meshlet_vertices.size()),
	        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	        VMA_MEMORY_USAGE_CPU_TO_GPU));

	Buffer meshlet_triangle_staging(
	    p_device,
	    BufferDesc(
	        sizeof(uint32_t),
	        static_cast<uint32_t>(m_meshlet_triangles.size()),
	        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	        VMA_MEMORY_USAGE_CPU_TO_GPU));

	Buffer meshlet_staging(
	    p_device,
	    BufferDesc(
	        sizeof(ShaderInterop::Meshlet),
	        static_cast<uint32_t>(m_meshlets.size()),
	        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	        VMA_MEMORY_USAGE_CPU_TO_GPU));

	Buffer meshlet_bound_staging(
	    p_device,
	    BufferDesc(
	        sizeof(ShaderInterop::MeshletBound),
	        static_cast<uint32_t>(m_meshlet_bounds.size()),
	        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	        VMA_MEMORY_USAGE_CPU_TO_GPU));

	std::memcpy(vertex_buffer_staging.Map(), m_vertices.data(), vertex_buffer_staging.GetSize());
	std::memcpy(index_staging.Map(), m_indices.data(), index_staging.GetSize());
	std::memcpy(meshlet_vertex_staging.Map(), m_meshlet_vertices.data(), meshlet_vertex_staging.GetSize());
	std::memcpy(meshlet_triangle_staging.Map(), m_meshlet_triangles.data(), meshlet_triangle_staging.GetSize());
	std::memcpy(meshlet_staging.Map(), m_meshlets.data(), meshlet_staging.GetSize());
	std::memcpy(meshlet_bound_staging.Map(), m_meshlet_bounds.data(), meshlet_bound_staging.GetSize());

	vertex_buffer_staging.Flush(vertex_buffer_staging.GetSize());
	index_staging.Flush(index_staging.GetSize());
	meshlet_vertex_staging.Flush(meshlet_vertex_staging.GetSize());
	meshlet_triangle_staging.Flush(meshlet_triangle_staging.GetSize());
	meshlet_staging.Flush(meshlet_staging.GetSize());
	meshlet_bound_staging.Flush(meshlet_bound_staging.GetSize());

	vertex_buffer_staging.Unmap();
	index_staging.Unmap();
	meshlet_vertex_staging.Unmap();
	meshlet_triangle_staging.Unmap();
	meshlet_staging.Unmap();
	meshlet_bound_staging.Unmap();

	{
		auto &cmd_buffer = p_device->RequestCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, VK_QUEUE_TRANSFER_BIT);
		cmd_buffer.Begin();
		cmd_buffer.CopyBuffer(BufferCopyInfo{&vertex_buffer_staging}, BufferCopyInfo{m_vertex_buffer.get()}, vertex_buffer_staging.GetSize());
		cmd_buffer.CopyBuffer(BufferCopyInfo{&index_staging}, BufferCopyInfo{m_index_buffer.get()}, index_staging.GetSize());
		cmd_buffer.CopyBuffer(BufferCopyInfo{&meshlet_vertex_staging}, BufferCopyInfo{m_meshlet_vertex_buffer.get()}, meshlet_vertex_staging.GetSize());
		cmd_buffer.CopyBuffer(BufferCopyInfo{&meshlet_triangle_staging}, BufferCopyInfo{m_meshlet_triangle_buffer.get()}, meshlet_triangle_staging.GetSize());
		cmd_buffer.CopyBuffer(BufferCopyInfo{&meshlet_staging}, BufferCopyInfo{m_meshlet_buffer.get()}, meshlet_staging.GetSize());
		cmd_buffer.CopyBuffer(BufferCopyInfo{&meshlet_bound_staging}, BufferCopyInfo{m_meshlet_bound_buffer.get()}, meshlet_bound_staging.GetSize());
		cmd_buffer.End();
		p_device->SubmitIdle(cmd_buffer, VK_QUEUE_TRANSFER_BIT);
	}

	AccelerationStructureDesc as_desc = {};

	as_desc.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

	as_desc.geometry.sType                                          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	as_desc.geometry.geometryType                                   = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	as_desc.geometry.geometry.triangles.sType                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	as_desc.geometry.geometry.triangles.vertexFormat                = VK_FORMAT_R32G32B32_SFLOAT;
	as_desc.geometry.geometry.triangles.vertexData.deviceAddress    = m_vertex_buffer->GetDeviceAddress();
	as_desc.geometry.geometry.triangles.maxVertex                   = static_cast<uint32_t>(m_vertices.size());
	as_desc.geometry.geometry.triangles.vertexStride                = sizeof(ShaderInterop::Vertex);
	as_desc.geometry.geometry.triangles.indexType                   = VK_INDEX_TYPE_UINT32;
	as_desc.geometry.geometry.triangles.indexData.deviceAddress     = m_index_buffer->GetDeviceAddress();
	as_desc.geometry.geometry.triangles.transformData.deviceAddress = 0;
	as_desc.geometry.geometry.triangles.transformData.hostAddress   = nullptr;

	as_desc.range_info.primitiveCount  = static_cast<uint32_t>(m_indices.size()) / 3;
	as_desc.range_info.primitiveOffset = 0;
	as_desc.range_info.firstVertex     = 0;
	as_desc.range_info.transformOffset = 0;

	{
		auto &cmd_buffer = p_device->RequestCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, VK_QUEUE_COMPUTE_BIT);
		cmd_buffer.Begin();
		m_bottom_level_acceleration_structure->Build(cmd_buffer, as_desc);
		cmd_buffer.End();
		p_device->SubmitIdle(cmd_buffer, VK_QUEUE_COMPUTE_BIT);
	}
}

const BoundingBox &Submesh::GetBoundingBox()
{
	return m_bounding_box;
}

Material *Submesh::GetMaterial()
{
	return m_material;
}

Buffer &Submesh::GetVertexBuffer()
{
	return *m_vertex_buffer;
}

Buffer &Submesh::GetIndexBuffer()
{
	return *m_index_buffer;
}

Buffer &Submesh::GetMeshletVertexBuffer()
{
	return *m_meshlet_vertex_buffer;
}

Buffer &Submesh::GetMeshletTriangleBuffer()
{
	return *m_meshlet_triangle_buffer;
}

Buffer &Submesh::GetMeshletBuffer()
{
	return *m_meshlet_buffer;
}

Buffer &Submesh::GetMeshletBoundBuffer()
{
	return *m_meshlet_bound_buffer;
}

AccelerationStructure &Submesh::GetBLAS()
{
	return *m_bottom_level_acceleration_structure;
}

Mesh::Mesh(RHIDevice *device) :
    p_device(device)
{
}

const std::string &Mesh::GetName() const
{
	return m_name;
}

void Mesh::UpdateBuffer()
{
	for (auto &submesh : m_submeshes)
	{
		submesh->UpdateBuffer();
	}
}

bool Mesh::OnImGui(ImGuiContext &context)
{
	bool     is_update  = false;
	uint32_t submesh_id = 0;
	for (auto &submesh : m_submeshes)
	{
		ImGui::PushID(submesh_id++);
		if (ImGui::TreeNode(submesh->GetName().c_str()))
		{
			if (submesh->OnImGui(context))
			{
				is_update = true;
			}
			ImGui::TreePop();
		}
		ImGui::PopID();
	}
	return is_update;
}

const std::vector<std::unique_ptr<Submesh>> &Mesh::GetSubmeshes() const
{
	return m_submeshes;
}

}        // namespace Ilum