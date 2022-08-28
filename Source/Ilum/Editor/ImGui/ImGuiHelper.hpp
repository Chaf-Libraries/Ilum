#pragma once

#include <imgui.h>

#include <rttr/type.h>

namespace ImGui
{
bool EditVariant(rttr::variant &var);
}        // namespace ImGui