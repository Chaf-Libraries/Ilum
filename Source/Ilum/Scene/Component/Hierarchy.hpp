#pragma once

#include "Component.hpp"

#include <entt.hpp>

#include <imgui.h>
#include <imgui_internal.h>

namespace Ilum::cmpt
{
struct Hierarchy : public Component
{
	entt::entity parent = entt::null;
	entt::entity first  = entt::null;
	entt::entity next   = entt::null;
	entt::entity prev   = entt::null;

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(parent, first, next, prev);
	}

	bool OnImGui(ImGuiContext &context);
};
}        // namespace Ilum::cmpt