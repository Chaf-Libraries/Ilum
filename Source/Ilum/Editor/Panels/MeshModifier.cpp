#include "MeshModifier.hpp"

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



	ImGui::End();
}
}        // namespace Ilum::panel