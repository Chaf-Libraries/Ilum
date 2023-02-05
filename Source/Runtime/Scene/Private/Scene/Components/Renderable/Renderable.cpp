#include "Components/Renderable/Renderable.hpp"

#include <Scene/Node.hpp>

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

bool Renderable::OnImGui()
{
	if (ImGui::TreeNode("Material"))
	{
		for (auto &material : m_materials)
		{
			ImGui::PushID(material.c_str());

			if (ImGui::Button(material.c_str(), ImVec2(ImGui::GetContentRegionAvail().x * 0.8f, 30.f)))
			{
				material = "";
				m_update = true;
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
					material = static_cast<const char *>(pay_load->Data);
					m_update = true;
				}
			}

			ImGui::PopID();
		}

		if (ImGui::Button("+"))
		{
			m_materials.emplace_back("");
			m_update = true;
		}

		ImGui::SameLine();

		if (ImGui::Button("-"))
		{
			m_materials.pop_back();
			m_update = true;
		}

		ImGui::TreePop();
	}

	return m_update;
}

void Renderable::Save(OutputArchive &archive) const
{
	archive(m_submeshes, m_materials, m_animations);
}

void Renderable::Load(InputArchive &archive)
{
	archive(m_submeshes, m_materials, m_animations);
	m_update = true;
}

void Renderable::AddMaterial(const std::string &material)
{
	m_materials.emplace_back(material);
	m_update = true;
}

void Renderable::AddSubmesh(const std::string &submesh)
{
	m_submeshes.emplace_back(submesh);
	m_update = true;
}

void Renderable::AddAnimation(const std::string &animation)
{
	m_animations.emplace_back(animation);
	m_update = true;
}

const std::vector<std::string> &Renderable::GetSubmeshes() const
{
	return m_submeshes;
}

const std::vector<std::string> &Renderable::GetMaterials() const
{
	return m_materials;
}

const std::vector<std::string> &Renderable::GetAnimations() const
{
	return m_animations;
}
}        // namespace Cmpt
}        // namespace Ilum