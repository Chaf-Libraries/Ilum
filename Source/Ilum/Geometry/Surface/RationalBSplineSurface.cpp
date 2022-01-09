#include "RationalBSplineSurface.hpp"

#include <tbb/tbb.h>

namespace Ilum::geometry
{
inline float gen_basis(uint32_t i, uint32_t k, float t)
{
	if (k == 1)
	{
		if (t >= i && t < i + 1)
		{
			return 1.f;
		}
		else
		{
			return 0.f;
		}
	}

	float left = 0.f;
	left       = static_cast<float>(k) - 1.f == 0.f ? 0.f : (t - static_cast<float>(i)) / (static_cast<float>(k) - 1.f) * gen_basis(i, k - 1, t);

	float right = 0.f;
	right       = static_cast<float>(k) - 1.f == 0.f ? 0.f : (static_cast<float>(i + k) - t) / (static_cast<float>(k) - 1.f) * gen_basis(i + 1, k - 1, t);

	return left + right;
}

void RationalBSplineSurface::generateVertices(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices, const std::vector<std::vector<glm::vec3>> &control_points, uint32_t sample_x, uint32_t sample_y)
{
	if (control_points.empty() || control_points[0].empty())
	{
		return;
	}

	vertices.resize((sample_x + 1) * (sample_y + 1));
	indices.resize(sample_x * sample_y * 2 * 3);

	tbb::parallel_for(tbb::blocked_range2d<uint32_t>(0, sample_x + 1u, 1, 0, sample_y + 1u, 1), [this, sample_x, sample_y, control_points, &vertices, &indices](const tbb::blocked_range2d<uint32_t> &count) {
		for (uint32_t i = count.rows().begin(); i != count.rows().end(); i++)
		{
			for (uint32_t j = count.cols().begin(); j != count.cols().end(); j++)
			{
				float u = static_cast<float>(control_points.size() + 1 - order) / static_cast<float>(sample_x) * static_cast<float>(i) + static_cast<float>(order) - 1.f;
				float v = static_cast<float>(control_points.size() + 1 - order) / static_cast<float>(sample_y) * static_cast<float>(j) + static_cast<float>(order) - 1.f;
				vertices[i * (sample_y + 1) + j].position = value(control_points, u, v);
				vertices[i * (sample_y + 1) + j].texcoord = glm::vec2(u, v);

				if (i >= sample_x || j >= sample_y)
				{
					continue;
				}

				uint32_t tri_idx     = (i * sample_y + j) * 2 * 3;
				indices[tri_idx]     = i * (sample_y + 1) + j;
				indices[tri_idx + 2] = i * (sample_y + 1) + j + 1;
				indices[tri_idx + 1] = (i + 1) * (sample_y + 1) + j + 1;
				indices[tri_idx + 3] = i * (sample_y + 1) + j;
				indices[tri_idx + 5] = (i + 1) * (sample_y + 1) + j + 1;
				indices[tri_idx + 4] = (i + 1) * (sample_y + 1) + j;
			}
		}
	});
}

glm::vec3 RationalBSplineSurface::value(const std::vector<std::vector<glm::vec3>> &control_points, float u, float v)
{
	uint32_t nu = static_cast<uint32_t>(control_points.size());
	uint32_t nv = static_cast<uint32_t>(control_points[0].size());

	float     denominator = 0.f;
	glm::vec3 numerator   = glm::vec3(0.f);

	for (uint32_t i = 0; i < nu; i++)
	{
		for (uint32_t j = 0; j < nv; j++)
		{
			float weight = gen_basis(i, order, u) * gen_basis(j, order, v) * weights[i][j];
			denominator += weight;
			numerator += weight * control_points[i][j];
		}
	}

	return numerator / denominator;
}
}        // namespace Ilum::geometry