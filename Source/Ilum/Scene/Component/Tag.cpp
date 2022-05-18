#include "Tag.hpp"

#include <imgui.h>

namespace Ilum::cmpt
{
bool Tag::OnImGui(ImGuiContext &context)
{
	bool update = false;
	ImGui::Text("Tag");

	ImGui::SameLine();
	char buffer[64];
	memset(buffer, 0, sizeof(buffer));
	std::memcpy(buffer, m_name.data(), sizeof(buffer));
	ImGui::PushItemWidth(150.f);
	if (ImGui::InputText("##Tag", buffer, sizeof(buffer)))
	{
		m_name = std::string(buffer);

		update = true;
	}
	ImGui::PopItemWidth();

	return update;
}

void Tag::SetName(const std::string &name)
{
	m_name = name;
}

const std::string &Tag::GetName() const
{
	return m_name;
}

}        // namespace Ilum::cmpt