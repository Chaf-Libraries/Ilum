#include "MeshModifier.hpp"

#include "Scene/Component/Renderable.hpp"

#include "Geometry/Mesh/Process/Subdivision.hpp"
#include "Geometry/Mesh/Process/Parameterization.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Editor/Editor.hpp"

#include <imgui.h>

namespace Ilum::panel
{
MeshModifier::MeshModifier()
{
	m_name = "Mesh Modifier";
}

void MeshModifier::draw(float delta_time)
{
	ImGui::Begin(m_name.c_str(), &active);

	auto entity = Editor::instance()->getSelect();
	if (!entity || !entity.hasComponent<cmpt::DynamicMeshRenderer>())
	{
		ImGui::End();
		return;
	}

	auto &dynamic_mesh = entity.getComponent<cmpt::DynamicMeshRenderer>();

	if (dynamic_mesh.vertices.empty())
	{
		ImGui::End();
		return;
	}

	if (ImGui::TreeNode("Subdivision"))
	{
		if (ImGui::Button("Loop Subdivision"))
		{
			auto [vertices, indices] = geometry::Subdivision::LoopSubdivision(dynamic_mesh.vertices, dynamic_mesh.indices);
			dynamic_mesh.vertices    = std::move(vertices);
			dynamic_mesh.indices     = std::move(indices);
			if (dynamic_mesh.vertices.size() * sizeof(Vertex) > dynamic_mesh.vertex_buffer.getSize() || dynamic_mesh.indices.size() * sizeof(uint32_t) > dynamic_mesh.index_buffer.getSize())
			{
				GraphicsContext::instance()->getQueueSystem().waitAll();
				dynamic_mesh.vertex_buffer = Buffer(dynamic_mesh.vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
				dynamic_mesh.index_buffer = Buffer(dynamic_mesh.indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			}
			std::memcpy(dynamic_mesh.vertex_buffer.map(), dynamic_mesh.vertices.data(), dynamic_mesh.vertices.size() * sizeof(Vertex));
			std::memcpy(dynamic_mesh.index_buffer.map(), dynamic_mesh.indices.data(), dynamic_mesh.indices.size() * sizeof(uint32_t));
			dynamic_mesh.vertex_buffer.unmap();
			dynamic_mesh.index_buffer.unmap();
		}
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Parameterization"))
	{
		if (ImGui::Button("Minimum Surface"))
		{
			auto [vertices, indices] = geometry::Parameterization::MinimumSurface(dynamic_mesh.vertices, dynamic_mesh.indices);
			dynamic_mesh.vertices    = std::move(vertices);
			dynamic_mesh.indices     = std::move(indices);
			if (dynamic_mesh.vertices.size() * sizeof(Vertex) > dynamic_mesh.vertex_buffer.getSize() || dynamic_mesh.indices.size() * sizeof(uint32_t) > dynamic_mesh.index_buffer.getSize())
			{
				GraphicsContext::instance()->getQueueSystem().waitAll();
				dynamic_mesh.vertex_buffer = Buffer(dynamic_mesh.vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
				dynamic_mesh.index_buffer  = Buffer(dynamic_mesh.indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			}
			std::memcpy(dynamic_mesh.vertex_buffer.map(), dynamic_mesh.vertices.data(), dynamic_mesh.vertices.size() * sizeof(Vertex));
			std::memcpy(dynamic_mesh.index_buffer.map(), dynamic_mesh.indices.data(), dynamic_mesh.indices.size() * sizeof(uint32_t));
			dynamic_mesh.vertex_buffer.unmap();
			dynamic_mesh.index_buffer.unmap();
		}

		ImGui::TreePop();
	}

	ImGui::End();
}
}        // namespace Ilum::panel