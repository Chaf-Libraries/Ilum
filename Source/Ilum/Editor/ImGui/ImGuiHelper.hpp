#pragma once

#include <imgui.h>

#include <rttr/type.h>

namespace ImGui
{
bool EditVariant_(const rttr::variant &var);

template <typename T>
inline bool EditVariant(T &var)
{
	rttr::variant rttr_var = var;
	bool          update   = EditVariant_(rttr_var);
	var                    = rttr_var.convert<T>();
	return update;
}

template <>
inline bool EditVariant<rttr::variant>(rttr::variant &var)
{
	return EditVariant_(var);
}

template <>
inline bool EditVariant<const rttr::variant>(const rttr::variant &var)
{
	return EditVariant_(var);
}

}        // namespace ImGui