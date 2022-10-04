#pragma once

#include "Component.hpp"

#include <vector>

namespace Ilum
{
STRUCT(StaticMeshComponent, Enable) :
    public Component
{
	size_t uuid;

	std::vector<std::string> materials;
};
}        // namespace Ilum