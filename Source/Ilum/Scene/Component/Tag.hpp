#pragma once

#include "Component.hpp"

#include <string>

namespace Ilum::cmpt
{
struct Tag : public Component
{
	std::string name = "Untitled Entity";

	Tag() = default;
	Tag(const std::string &name) :
	    name(name)
	{}

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(name);
	}

	bool OnImGui(ImGuiContext &context) override;
};
}        // namespace Ilum::cmpt