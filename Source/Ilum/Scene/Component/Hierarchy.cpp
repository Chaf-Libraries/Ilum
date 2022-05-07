#include "Hierarchy.hpp"

namespace Ilum::cmpt
{
bool Hierarchy::OnImGui(ImGuiContext &context)
{
	ImGui::Text("Parent: %s", parent == entt::null ? "false" : "true");
	ImGui::Text("Children: %s", first == entt::null ? "false" : "true");
	ImGui::Text("Siblings: %s", next == entt::null && prev == entt::null ? "false" : "true");
	return false;
}

}        // namespace Ilum::cmpt