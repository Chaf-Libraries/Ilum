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

	STRUCT(MaterialID, Enable)
	{
		META(DragDrop("Material"), NAME(""))
		size_t material_id=100;
	};

	std::vector<MaterialID> materials;
};
}        // namespace Ilum