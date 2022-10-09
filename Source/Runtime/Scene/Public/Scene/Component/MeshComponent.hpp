#pragma once

#include "Component.hpp"
#include "Precompile.hpp"

#include <vector>

namespace Ilum
{
STRUCT(MeshComponent, Enable) :
    public Component{};

STRUCT(StaticMeshComponent, Enable) :
    public MeshComponent
{
	size_t uuid;

	std::vector<std::string> materials;
};
}        // namespace Ilum