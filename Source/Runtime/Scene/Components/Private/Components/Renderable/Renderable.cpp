#include "Renderable/Renderable.hpp"

#include <SceneGraph/Node.hpp>

#include <imgui.h>
#include <imgui_internal.h>

namespace Ilum
{
namespace Cmpt
{
Renderable::Renderable(const char *name, Node *node) :
    Component(name, node)
{
}

void Renderable::OnImGui()
{
	if (ImGui::TreeNode("Material"))
	{
		for (auto &material : m_materials)
		{
			ImGui::PushID(material.c_str());

			if (ImGui::Button(material.c_str(), ImVec2(ImGui::GetContentRegionAvail().x * 0.8f, 30.f)))
			{
				material = "";
			}

			if (ImGui::BeginDragDropSource())
			{
				ImGui::SetDragDropPayload("Material", material.c_str(), material.length() + 1);
				ImGui::EndDragDropSource();
			}

			if (ImGui::BeginDragDropTarget())
			{
				if (const auto *pay_load = ImGui::AcceptDragDropPayload("Material"))
				{
					material = *static_cast<std::string *>(pay_load->Data);
				}
			}

			ImGui::PopID();
		}

		if (ImGui::Button("+"))
		{
			m_materials.emplace_back("");
		}

		ImGui::SameLine();

		if (ImGui::Button("-"))
		{
			m_materials.pop_back();
		}

		ImGui::TreePop();
	}
}

void Renderable::AddMaterial(const std::string &material)
{
	m_materials.emplace_back(material);
}

void Renderable::AddSubmesh(const std::string &submesh)
{
	m_submeshes.emplace_back(submesh);
}
}        // namespace Cmpt
}        // namespace Ilum