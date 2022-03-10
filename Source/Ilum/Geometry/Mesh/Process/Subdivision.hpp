#pragma once

#include "IMeshProcess.hpp"

namespace Ilum::geometry
{
class Subdivision : public IMeshProcess
{
  public:
	static std::pair<std::vector<Vertex>, std::vector<uint32_t>> LoopSubdivision(const std::vector<Vertex> &in_vertices, const std::vector<uint32_t> &in_indices);
};
}        // namespace Ilum::geometry