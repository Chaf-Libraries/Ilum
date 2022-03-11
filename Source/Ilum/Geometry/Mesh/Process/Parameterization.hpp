#pragma once

#include "IMeshProcess.hpp"

namespace Ilum::geometry
{
class Parameterization : public IMeshProcess
{
  public:
	enum class TutteWeightType
	{
		Uniform,
		Cotangent,
		ShapePreserving
	};

	static std::pair<std::vector<Vertex>, std::vector<uint32_t>> MinimumSurface(const std::vector<Vertex> &in_vertices, const std::vector<uint32_t> &in_indices);

	static std::pair<std::vector<Vertex>, std::vector<uint32_t>> TutteParameterization(const std::vector<Vertex> &in_vertices, const std::vector<uint32_t> &in_indices, TutteWeightType weight_type);
};
}        // namespace Ilum::geometry