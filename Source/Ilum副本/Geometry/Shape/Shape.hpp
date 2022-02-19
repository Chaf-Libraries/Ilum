#pragma once

#include "Geometry/Mesh/TriMesh.hpp"

namespace Ilum::geometry
{
struct Shape
{
	virtual TriMesh toTriMesh()
	{
		return TriMesh();
	}
};
}