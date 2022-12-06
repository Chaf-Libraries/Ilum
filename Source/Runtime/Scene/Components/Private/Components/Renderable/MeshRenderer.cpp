#include "Renderable/MeshRenderer.hpp"
#include <SceneGraph/Node.hpp>

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
		for (auto &uuid : m_submeshes)
		{
			ImGui::PushID(static_cast<int32_t>(uuid));
			ImGui::Button(std::to_string(uuid).c_str(), ImVec2(200.f, 5.f));
			ImGui::PopID();
		}

		ImGui::TreePop();
	}
}

std::type_index MeshRenderer::GetType() const
{
	return typeid(MeshRenderer);
}
}        // namespace Cmpt
}        // namespace Ilum