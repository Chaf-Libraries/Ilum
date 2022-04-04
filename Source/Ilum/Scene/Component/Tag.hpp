#pragma once

#include <string>

#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>

namespace Ilum::cmpt
{
struct Tag
{
	std::string name = "Untitled Entity";

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(name);
	}
};
}        // namespace Ilum::cmpt