#pragma once

#include <string>

namespace Ilum::cmpt
{
struct Tag
{
	std::string name   = "Untitled Entity";
	bool        active = true;

	inline static bool update = false;
};
}        // namespace Ilum::Cmpt