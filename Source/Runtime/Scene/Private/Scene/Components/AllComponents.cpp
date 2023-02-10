#include "Components/AllComponents.hpp"

#include <imgui.h>

namespace Ilum
{
namespace Cmpt
{
void SetImGuiContext(ImGuiContext *context)
{
	return ImGui::SetCurrentContext(context);
}
}        // namespace Cmpt
}        // namespace Ilum
