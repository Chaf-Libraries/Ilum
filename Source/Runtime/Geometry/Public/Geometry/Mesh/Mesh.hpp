#pragma once

#include "../Precompile.hpp"

#include <glm/glm.hpp>

namespace Ilum
{
struct VertexData
{
	glm::vec3 position = glm::vec3(0.f);
	glm::vec3 normal   = glm::vec3(0.f);
	glm::vec2 uv       = glm::vec2(0.f);
};

struct  TriMesh
{
	std::vector<VertexData> vertices;
	std::vector<uint32_t>   indices;

	void GenerateNormal();
};

class  Mesh
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

	virtual TriMesh ToTriMesh() const = 0;

  protected:
	float Area(const glm::vec3 &v1, const glm::vec3 &v2, const glm::vec3 &v3) const;

	float LocalAverageLegion(const glm::vec3 &center, const std::vector<glm::vec3> &neighbors, LocalAverageLegionOption option = LocalAverageLegionOption::MixVoronoiCell) const;

	// neighbor points must in order
	glm::vec3 Laplace(const glm::vec3 &center, const std::vector<glm::vec3> &neighbors, LaplaceOption option = LaplaceOption::CotangentFormula) const;

	glm::vec3 Normal(const glm::vec3 &v1, const glm::vec3 &v2, const glm::vec3 &v3) const;

	glm::vec3 Normal(const glm::vec3 &center, const std::vector<glm::vec3> &neighbors, VertexNormalOption option = VertexNormalOption::Angle) const;

	float Curvature(const glm::vec3 &center, const std::vector<glm::vec3> &neighbors, CurvatureOption option = CurvatureOption::Gaussian);

  protected:
	std::pmr::unsynchronized_pool_resource m_pool;
};

}        // namespace Ilum