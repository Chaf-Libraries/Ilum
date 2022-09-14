#pragma once

#include <imgui.h>

#include <rttr/type.h>

namespace ImGui
{
bool EditVariantImpl(const rttr::variant &var);

template <typename T>
inline bool EditVariant(T &var)
{
	rttr::variant rttr_var = var;
	bool          update   = EditVariantImpl(rttr_var);
	var                    = rttr_var.convert<T>();
	return update;
}

template <>
inline bool EditVariant<rttr::variant>(rttr::variant &var)
{
	return EditVariantImpl(var);
}

template <>
inline bool EditVariant<const rttr::variant>(const rttr::variant &var)
{
	return EditVariantImpl(var);
}

}        // namespace ImGui