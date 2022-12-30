#include "Components/Renderable/MeshRenderer.hpp"
#include <Scene/Node.hpp>

#include <imgui.h>

namespace Ilum
{
namespace Cmpt
{
MeshRenderer::MeshRenderer(Node *node) :
    Renderable("Mesh Renderer", node)
{
}

void MeshRenderer::OnImGui()
{
	if (ImGui::TreeNode("Submesh"))
	{
		for (auto &submesh : m_submeshes)
		{
			ImGui::PushID(submesh.c_str());

			if (ImGui::Button(submesh.c_str(), ImVec2(ImGui::GetContentRegionAvail().x * 0.8f, 30.f)))
			{
				submesh = "";
			}

			if (ImGui::BeginDragDropSource())
			{
				ImGui::SetDragDropPayload("Mesh", submesh.c_str(), submesh.length() + 1);
				ImGui::EndDragDropSource();
			}

			if (ImGui::BeginDragDropTarget())
			{
				if (const auto *pay_load = ImGui::AcceptDragDropPayload("Mesh"))
				{
					submesh = static_cast<const char *>(pay_load->Data);
				}
			}

			ImGui::PopID();
		}

		if (ImGui::Button("+"))
		{
			m_submeshes.emplace_back("");
		}

		ImGui::SameLine();

		if (ImGui::Button("-"))
		{
			m_submeshes.pop_back();
		}

		ImGui::TreePop();
	}

	Renderable::OnImGui();
}

std::type_index MeshRenderer::GetType() const
{
	return typeid(MeshRenderer);
}
}        // namespace Cmpt
}        // namespace Ilum