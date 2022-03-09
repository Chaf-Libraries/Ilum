#pragma once

#include "Geometry/Vertex.hpp"

namespace Ilum::geometry
{
struct Shape
{
	virtual std::pair<std::vector<Vertex>, std::vector<uint32_t>> toMesh() = 0;
};
}