#include "RationalBezier.hpp"

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

std::vector<glm::vec3> RationalBezier::generateVertices(const std::vector<glm::vec3> &control_points, uint32_t sample)
{
	if (control_points.empty())
	{
		return {};
	}

	std::vector<glm::vec3> result((size_t) sample + 1);

	tbb::parallel_for(tbb::blocked_range<uint32_t>(0, sample + 1u), [this, sample, control_points, &result](const tbb::blocked_range<uint32_t> &count) {
		for (uint32_t k = count.begin(); k != count.end(); k++)
		{
			float t = static_cast<float>(k) / static_cast<float>(sample);

			result[k] = value(control_points, t);
		}
	});

	return result;
}

glm::vec3 RationalBezier::value(const std::vector<glm::vec3> &control_points, float t)
{
	uint32_t n = static_cast<uint32_t>(control_points.size());

	float     denominator = 0.f;
	glm::vec3 numerator   = glm::vec3(0.f);

	for (uint32_t i = 0; i < n; i++)
	{
		float weight = bernstein_basis(n - 1, i, t) * weights[i];
		denominator += weight;
		numerator += weight * control_points[i];
	}

	return numerator / denominator;
}
}        // namespace Ilum::geometry
