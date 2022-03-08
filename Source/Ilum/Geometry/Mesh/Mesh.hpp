#pragma once

#include "Utils/PCH.hpp"

#include <glm/glm.hpp>

namespace Ilum::geometry
{
class Mesh
{
  public:
	enum class VertexNormalOption
	{
		Uniform,
		Area,
		Angle
	};

	enum class LaplaceOption
	{
		Uniform,
		CotangentFormula
	};

	enum class CurvatureOption
	{
		Mean,
		AbsoluteMean,
		Gaussian
	};

	enum class LocalAverageLegionOption
	{
		BarycentricCell,
		VoronoiCell,
		MixVoronoiCell
	};

  public:
	Mesh() = default;

	~Mesh() = default;

	virtual std::pair<std::vector<glm::vec3>, std::vector<uint32_t>> toTriMesh() const = 0;

  protected:
	float area(const glm::vec3 &v1, const glm::vec3 &v2, const glm::vec3 &v3) const;

	float localAverageLegion(const glm::vec3 &center, const std::vector<glm::vec3> &neighbors, LocalAverageLegionOption option = LocalAverageLegionOption::MixVoronoiCell) const;

	// neighbor points must in order
	glm::vec3 laplace(const glm::vec3 &center, const std::vector<glm::vec3> &neighbors, LaplaceOption option = LaplaceOption::CotangentFormula) const;

	glm::vec3 normal(const glm::vec3 &v1, const glm::vec3 &v2, const glm::vec3 &v3) const;

	glm::vec3 normal(const glm::vec3 &center, const std::vector<glm::vec3> &neighbors, VertexNormalOption option = VertexNormalOption::Angle) const;

	float curvature(const glm::vec3 &center, const std::vector<glm::vec3> &neighbors, CurvatureOption option = CurvatureOption::Gaussian);

  protected:
	std::pmr::unsynchronized_pool_resource m_pool;
};
}        // namespace Ilum::geometry