#pragma once

#include "Component.hpp"

namespace Ilum
{
STRUCT(TagComponent, Enable) :
    public Component
{
	std::string tag;
};
}        // namespace Ilum