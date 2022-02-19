#pragma once

#include <meshoptimizer.h>

#include <cstdint>

namespace Ilum::Resource
{
struct Meshlet
{
	meshopt_Bounds bounds;
	uint32_t       indices_offset;
	uint32_t       indices_count;
	uint32_t       vertices_offset;
};
}