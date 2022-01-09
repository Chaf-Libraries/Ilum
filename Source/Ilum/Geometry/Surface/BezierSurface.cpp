#include "BezierSurface.hpp"

#include <tbb/tbb.h>

namespace Ilum::geometry
{
inline uint32_t Cnk(uint32_t n, uint32_t k)
{
	if (n == k || k == 0)
	{
		return 1;
	}

	std::vector<uint32_t> dp(k + 1u);
	for (uint32_t i = 0; i <= n; i++)
	{
		for (int64_t j = static_cast<int64_t>(std::min(i, k)); j >= 0; j--)
		{
			if (i == j || j == 0)
			{
				dp[j] = 1;
			}
			else
			{
				dp[j] = dp[j] + dp[j - 1];
			}
		}
	}

	return dp[k];
}

inline float bernstein_basis(uint32_t n, uint32_t i, float t)
{
	return static_cast<float>(Cnk(n, i)) * static_cast<float>(std::pow(t, i) * std::pow(1 - t, n - i));
}

void BezierSurface::generateVertices(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices, const std::vector<std::vector<glm::vec3>> &control_points, uint32_t sample_x, uint32_t sample_y)
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
				float u = static_cast<float>(i) / static_cast<float>(sample_x);
				float v = static_cast<float>(j) / static_cast<float>(sample_y);
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

glm::vec3 BezierSurface::value(const std::vector<std::vector<glm::vec3>> &control_points, float u, float v)
{
	uint32_t nu = static_cast<uint32_t>(control_points.size());
	uint32_t nv = static_cast<uint32_t>(control_points[0].size());

	glm::vec3 result = glm::vec3(0.f);

	for (uint32_t i = 0; i < nu; i++)
	{
		for (uint32_t j = 0; j < nv; j++)
		{
			result += bernstein_basis(nu - 1, i, u) * bernstein_basis(nv - 1, j, v) * control_points[i][j];
		}
	}

	return result;
}
}        // namespace Ilum::geometry