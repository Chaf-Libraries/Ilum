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
	META(DragDrop("Model"))
	size_t uuid;

	std::vector<size_t> materials;
};
}        // namespace Ilum