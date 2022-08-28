#pragma once

#include <imgui.h>

#include <rttr/type.h>

namespace ImGui
{
bool EditVariant(rttr::variant &var);

template<typename T>
bool EditVariant(T& var)
{
	rttr::variant rttr_var = var;
	bool update = EditVariant(rttr_var);
	var = rttr_var.convert<T>();
	return update;
}
}        // namespace ImGui