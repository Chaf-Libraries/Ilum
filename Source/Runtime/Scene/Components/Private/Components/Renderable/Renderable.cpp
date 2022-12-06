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

//void Renderable::OnImGui()
//{
//	if (ImGui::TreeNode("Submesh"))
//	{
//		for (auto &uuid : m_submeshes)
//		{
//			ImGui::PushID(static_cast<int32_t>(uuid));
//			ImGui::Button(std::to_string(uuid).c_str(), ImVec2(200.f, 5.f));
//			ImGui::PopID();
//		}
//
//		ImGui::TreePop();
//	}
//}

void Renderable::AddMaterial(size_t uuid)
{
	m_materials.emplace_back(uuid);
}

void Renderable::AddSubmesh(size_t uuid)
{
	m_submeshes.emplace_back(uuid);
}
}        // namespace Cmpt
}        // namespace Ilum