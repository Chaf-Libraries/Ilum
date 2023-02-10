#include "Mesh/Mesh.hpp"

#include <glm/gtc/constants.hpp>

namespace Ilum
{
void TriMesh::GenerateNormal()
{
	for (size_t i = 0; i < vertices.size(); i++)
	{
		vertices[i].normal = glm::vec3(0.f);
	}

	for (size_t i = 0; i < indices.size(); i += 3)
	{
		glm::vec3 e2 = vertices[indices[i + 1]].position - vertices[indices[i]].position;
		glm::vec3 e1 = vertices[indices[i]].position - vertices[indices[i + 2]].position;

		glm::vec3 normal = glm::normalize(glm::cross(e1, e2));
		vertices[indices[i]].normal += normal;
		vertices[indices[i + 1]].normal += normal;
		vertices[indices[i + 2]].normal += normal;
	}

	for (size_t i = 0; i < vertices.size(); i++)
	{
		vertices[i].normal = glm::normalize(vertices[i].normal);
	}
}

float Mesh::Area(const glm::vec3 &v1, const glm::vec3 &v2, const glm::vec3 &v3) const
{
	glm::vec3 e1 = v2 - v1;
	glm::vec3 e2 = v3 - v1;

	return glm::length(glm::cross(e1, e2));
}

float Mesh::LocalAverageLegion(const glm::vec3 &center, const std::vector<glm::vec3> &neighbors, LocalAverageLegionOption option) const
{
	if (option == LocalAverageLegionOption::BarycentricCell)
	{
		float local_average_legion = 0.f;
		for (size_t i = 0; i < neighbors.size() - 1; i++)
		{
			local_average_legion += Area(center, neighbors[i], neighbors[i + 1]) / 3.f;
		}
		return local_average_legion;
	}
	else if (option == LocalAverageLegionOption::VoronoiCell)
	{
		float local_average_legion = 0.f;
		for (size_t i = 0; i < neighbors.size() - 1; i++)
		{
			glm::vec3 e1 = neighbors[i] - center;
			glm::vec3 e2 = neighbors[i + 1] - center;
			glm::vec3 n  = Normal(center, neighbors[i], neighbors[i + 1]);

			glm::vec3 vb1 = glm::normalize(glm::cross(n, e1));
			glm::vec3 vb2 = glm::normalize(glm::cross(e2, n));

			glm::vec3 mp1 = (neighbors[i] + center) * 0.5f;
			glm::vec3 mp2 = (neighbors[i + 1] + center) * 0.5f;

			float     t = ((vb1.y * mp1.x - vb1.x * mp1.y) - (vb1.y * mp2.x - vb1.x * mp2.y)) / (vb1.y * vb2.x - vb1.x * vb2.y);
			glm::vec3 p = mp2 + vb2 * t;

			local_average_legion += Area(p, mp1, center) + Area(p, mp2, center);
		}
		return local_average_legion;
	}
	else if (option == LocalAverageLegionOption::MixVoronoiCell)
	{
		float local_average_legion = 0.f;
		for (size_t i = 0; i < neighbors.size() - 1; i++)
		{
			glm::vec3 e1 = neighbors[i] - center;
			glm::vec3 e2 = neighbors[i + 1] - center;

			if (glm::dot(e1, e2) < 0.f)
			{
				glm::vec3 mp  = (neighbors[i] + neighbors[i + 1]) * 0.5f;
				glm::vec3 mp1 = (neighbors[i] + center) * 0.5f;
				glm::vec3 mp2 = (neighbors[i + 1] + center) * 0.5f;

				local_average_legion += Area(mp, mp1, center) + Area(mp, mp2, center);
			}
			else
			{
				glm::vec3 n = Normal(center, neighbors[i], neighbors[i + 1]);

				glm::vec3 vb1 = glm::normalize(glm::cross(n, e1));
				glm::vec3 vb2 = glm::normalize(glm::cross(e2, n));

				glm::vec3 mp1 = (neighbors[i] + center) * 0.5f;
				glm::vec3 mp2 = (neighbors[i + 1] + center) * 0.5f;

				float     t = ((vb1.y * mp1.x - vb1.x * mp1.y) - (vb1.y * mp2.x - vb1.x * mp2.y)) / (vb1.y * vb2.x - vb1.x * vb2.y);
				glm::vec3 p = mp2 + vb2 * t;

				local_average_legion += Area(p, mp1, center) + Area(p, mp2, center);
			}
		}
		return local_average_legion;
	}

	return 0.0f;
}

glm::vec3 Mesh::Laplace(const glm::vec3 &center, const std::vector<glm::vec3> &neighbors, LaplaceOption option) const
{
	if (option == LaplaceOption::Uniform)
	{
		glm::vec3 result = glm::vec3(0.f);
		for (auto &neighbor : neighbors)
		{
			result += (neighbor - center);
		}
		return result / static_cast<float>(neighbors.size());
	}
	else if (option == LaplaceOption::CotangentFormula)
	{
		glm::vec3 result = glm::vec3(0.f);
		for (size_t i = 0; i < neighbors.size(); i++)
		{
			glm::vec3 current = neighbors[i];
			glm::vec3 next    = neighbors[(i + 1) % neighbors.size()];
			glm::vec3 prev    = neighbors[(i + neighbors.size() - 1) % neighbors.size()];

			float cos_alpha = glm::dot(center - next, current - next) / (glm::length(center - next), glm::length(current - next));
			float cos_beta  = glm::dot(center - prev, current - prev) / (glm::length(center - prev), glm::length(current - prev));

			float cot_alpha = cos_alpha / glm::sqrt(1 - cos_alpha * cos_alpha);
			float cot_beta  = cos_beta / glm::sqrt(1 - cos_beta * cos_beta);

			result += (cot_alpha + cot_beta) * (current - center);
		}
		return result / (2.f * LocalAverageLegion(center, neighbors));
	}

	return glm::vec3(0.f);
}

glm::vec3 Mesh::Normal(const glm::vec3 &v1, const glm::vec3 &v2, const glm::vec3 &v3) const
{
	glm::vec3 e1 = v2 - v1;
	glm::vec3 e2 = v3 - v1;

	return glm::normalize(glm::cross(e1, e2));
}

glm::vec3 Mesh::Normal(const glm::vec3 &center, const std::vector<glm::vec3> &neighbors, VertexNormalOption option) const
{
	if (option == VertexNormalOption::Uniform)
	{
		glm::vec3 norm = glm::vec3(0.f);
		for (size_t i = 0; i < neighbors.size() - 1; i++)
		{
			norm += Normal(center, neighbors[i], neighbors[i + 1]);
		}
		return glm::normalize(norm);
	}
	else if (option == VertexNormalOption::Area)
	{
		glm::vec3 norm = glm::vec3(0.f);
		for (size_t i = 0; i < neighbors.size() - 1; i++)
		{
			norm += Area(center, neighbors[i], neighbors[i + 1]) * Normal(center, neighbors[i], neighbors[i + 1]);
		}
		return glm::normalize(norm);
	}
	else if (option == VertexNormalOption::Angle)
	{
		glm::vec3 norm = glm::vec3(0.f);
		for (size_t i = 0; i < neighbors.size() - 1; i++)
		{
			glm::vec3 e1    = neighbors[i] - center;
			glm::vec3 e2    = neighbors[i + 1] - center;
			float     angle = glm::acos(glm::dot(e1, e2) / (glm::length(e1) * glm::length(e2)));
			norm += angle * Normal(center, neighbors[i], neighbors[i + 1]);
		}
		return glm::normalize(norm);
	}

	return glm::vec3(0.f);
}

float Mesh::Curvature(const glm::vec3 &center, const std::vector<glm::vec3> &neighbors, CurvatureOption option)
{
	if (option == CurvatureOption::Mean)
	{
		glm::vec3 laplacian = Laplace(center, neighbors);
		glm::vec3 norm      = Normal(center, neighbors);
		return -0.5f * glm::dot(laplacian, norm);
	}
	else if (option == CurvatureOption::AbsoluteMean)
	{
		glm::vec3 laplacian = Laplace(center, neighbors);
		glm::vec3 norm      = Normal(center, neighbors);
		return glm::abs(-0.5f * glm::dot(laplacian, norm));
	}
	else if (option == CurvatureOption::Gaussian)
	{
		float result = 2.f * glm::pi<float>();
		for (size_t i = 0; i < neighbors.size() - 1; i++)
		{
			glm::vec3 e1    = neighbors[i] - center;
			glm::vec3 e2    = neighbors[i + 1] - center;
			float     angle = glm::acos(glm::dot(e1, e2) / (glm::length(e1) * glm::length(e2)));
			result -= angle;
		}
		return result / LocalAverageLegion(center, neighbors);
	}

	return 0.0f;
}
}        // namespace Ilum